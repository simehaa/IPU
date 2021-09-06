#include <chrono>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include "utils.hpp"

poplar::ComputeSet createComputeSet(
  poplar::Graph &graph,
  poplar::Tensor &e_in,
  poplar::Tensor &e_out,
  poplar::Tensor &r_in,
  poplar::Tensor &r_out,
  utils::Options &options,
  const std::string& compute_set_name,
  const std::string& vertex) {

  auto compute_set = graph.addComputeSet(compute_set_name);
  unsigned num_workers_per_tile = 1; //graph.getTarget().getNumWorkerContexts();
  unsigned nh = options.nh; // Number of splits in height
  unsigned nw = options.nw; // Number of splits in width

  // this double loop essentially iterated over the number of tiles (see tile_id below)
  for (std::size_t x = 0; x < nh; ++x) {
    for (std::size_t y = 0; y < nw; ++y) {

      // tile_id should run from 0, 1, ..., nh*nw
      unsigned tile_id = index(x, y, nw);

      unsigned tile_x = block_low(x, nh, options.height) + 1;
      unsigned tile_height = block_size(x, nh, options.height);
      unsigned y_low = block_low(y, nw, options.width) + 1;
      unsigned y_high = block_high(y, nw, options.width) + 1;

      for (std::size_t worker_i = 0; worker_i < num_workers_per_tile; ++worker_i) {
        
        // Dividing tile work among workers by splitting up grid further in height (x) dimension
        unsigned x_low = tile_x + block_low(worker_i, num_workers_per_tile, tile_height);
        unsigned x_high = tile_x + block_high(worker_i, num_workers_per_tile, tile_height);

        // Make slices
        auto e_in_slice = e_in.slice({x_low-1, y_low-1}, {x_high+1, y_high+1});
        auto r_in_slice = r_in.slice({x_low, y_low}, {x_high, y_high});
        auto e_out_slice = e_out.slice({x_low, y_low}, {x_high, y_high});
        auto r_out_slice = r_out.slice({x_low, y_low}, {x_high, y_high});

        std::vector<std::size_t> shape = e_out_slice.shape();
        if (area(shape) < area(options.smallest_slice))
          options.smallest_slice = shape;
        if (area(shape) > area(options.largest_slice)) 
          options.largest_slice = shape;

        // Assign vertex to graph
        auto v = graph.addVertex(compute_set, vertex);
        graph.connect(v["e_in"], e_in_slice);
        graph.connect(v["r_in"], r_in_slice);
        graph.connect(v["e_out"], e_out_slice);
        graph.connect(v["r_out"], r_out_slice);
        graph.setInitialValue(v["height"], shape[0]);
        graph.setInitialValue(v["width"], shape[1]);
        graph.setInitialValue(v["delta"], options.delta);
        graph.setInitialValue(v["epsilon"], options.epsilon);
        graph.setInitialValue(v["my1"], options.my1);
        graph.setInitialValue(v["my2"], options.my2);
        graph.setInitialValue(v["h"], options.h);
        graph.setInitialValue(v["dt"], options.dt);
        graph.setInitialValue(v["k"], options.k);
        graph.setInitialValue(v["a"], options.a);
        graph.setInitialValue(v["b"], options.b);
        graph.setTileMapping(v, tile_id);
      }
    }
  }

  return compute_set;
}

std::vector<poplar::program::Program> createIpuPrograms(
  poplar::Graph &graph,
  utils::Options &options) { 

  // Allocate Tensors, device variables
  long unsigned h = options.height+2;
  long unsigned w = options.width+2;
  auto e_a = graph.addVariable(poplar::FLOAT, {h, w}, "e_a");
  auto e_b = graph.addVariable(poplar::FLOAT, {h, w}, "e_b");
  auto r_a = graph.addVariable(poplar::FLOAT, {h, w}, "r_a");
  auto r_b = graph.addVariable(poplar::FLOAT, {h, w}, "r_b");

  // Tile mappings
  int nh = options.nh; // Number of splits in height
  int nw = options.nw; // Number of splits in width

  for (std::size_t x = 0; x < nh; ++x) {
    for (std::size_t y = 0; y < nw; ++y) {

      // tile_id runs from 0, 1, 2, ..., nh*nw
      unsigned tile_id = index(x, y, nw);

      // Evaluate offsets in all dimensions (avoid overlap at edges)
      std::size_t offset_top = (x == 0) ? 0 : 1;
      std::size_t offset_left = (y == 0) ? 0 : 1;
      std::size_t offset_bottom = (x == nh - 1) ? 2 : 1;
      std::size_t offset_right = (y == nw - 1) ? 2 : 1;
      std::size_t x_low = block_low(x, nh, h-2) + offset_top;
      std::size_t y_low = block_low(y, nw, w-2) + offset_left;
      std::size_t x_high = block_high(x, nh, h-2) + offset_bottom;
      std::size_t y_high = block_high(y, nw, w-2) + offset_right;

      // map a slice to a tile
      graph.setTileMapping(
        e_a.slice(
          {x_low,y_low},
          {x_high,y_high}
        ),
        tile_id
      );
    }
  }

  // Apply the tile mapping to all tensors
  const auto& tile_mapping = graph.getTileMapping(e_a);
  graph.setTileMapping(e_b, tile_mapping);
  graph.setTileMapping(r_a, tile_mapping);
  graph.setTileMapping(r_b, tile_mapping);

  // Define data streams
  long unsigned area = (h-2)*(w-2);
  auto host_to_device_e = graph.addHostToDeviceFIFO("host_to_device_stream_e", poplar::FLOAT, area);
  auto host_to_device_r = graph.addHostToDeviceFIFO("host_to_device_stream_r", poplar::FLOAT, area);
  auto device_to_host_e = graph.addDeviceToHostFIFO("device_to_host_stream_e", poplar::FLOAT, area);
  auto device_to_host_r = graph.addDeviceToHostFIFO("device_to_host_stream_r", poplar::FLOAT, area);

  std::vector<poplar::program::Program> programs;

  // Program 0: move content of initial_values into all four tensors
  programs.push_back(
    poplar::program::Sequence{
      poplar::program::Copy(host_to_device_e, e_a.slice({1,1},{h-1,w-1})),
      poplar::program::Copy(host_to_device_r, r_a.slice({1,1},{h-1,w-1})),
      poplar::program::Copy(e_a.slice({1,1},{h-1,w-1}), e_b.slice({1,1},{h-1,w-1})),
      poplar::program::Copy(r_a.slice({1,1},{h-1,w-1}), r_b.slice({1,1},{h-1,w-1})),
    }
  );

  // Program 1: execution(s) of compute sets
  poplar::program::Sequence execute_this_compute_set;
  auto compute_set_b_to_a = createComputeSet(graph, e_b, e_a, r_b, r_a, options, "Aliev_Panfilov_b_to_a", "AlievPanfilov");
  auto compute_set_a_to_b = createComputeSet(graph, e_a, e_b, r_a, r_b, options, "Aliev_Panfilov_a_to_b", "AlievPanfilov");
  
  poplar::program::Sequence a_to_b = {
    poplar::program::Copy(e_a.slice({2,1},{3,w-1}), e_a.slice({0,1},{1,w-1})), // north
    poplar::program::Copy(e_a.slice({h-3,1},{h-2,w-1}), e_a.slice({h-1,1},{h,w-1})), // south
    poplar::program::Copy(e_a.slice({1,2},{h-1,3}), e_a.slice({1,0},{h-1,1})), // west
    poplar::program::Copy(e_a.slice({1,w-3},{h-1,w-2}), e_a.slice({1,w-1},{h-1,w})), // east
    poplar::program::Execute(compute_set_a_to_b)
  };

  poplar::program::Sequence b_to_a = {
    poplar::program::Copy(e_b.slice({2,1},{3,w-1}), e_b.slice({0,1},{1,w-1})), // north
    poplar::program::Copy(e_b.slice({h-3,1},{h-2,w-1}), e_b.slice({h-1,1},{h,w-1})), // south
    poplar::program::Copy(e_b.slice({1,2},{h-1,3}), e_b.slice({1,0},{h-1,1})), // west
    poplar::program::Copy(e_b.slice({1,w-3},{h-1,w-2}), e_b.slice({1,w-1},{h-1,w})), // east
    poplar::program::Execute(compute_set_b_to_a)
  };

  if (options.num_iterations % 2 == 1) // if num_iterations is odd: add one extra iteration
    execute_this_compute_set.add(a_to_b);

  // add iterations 
  execute_this_compute_set.add(
    poplar::program::Repeat(
      options.num_iterations/2,
      poplar::program::Sequence{b_to_a, a_to_b}
    )
  );
  programs.push_back(execute_this_compute_set);

  // Program 2: copy final results back to host
  programs.push_back(
    poplar::program::Sequence{
      poplar::program::Copy(e_b.slice({1,1},{h-1,w-1}), device_to_host_e),
      poplar::program::Copy(r_b.slice({1,1},{h-1,w-1}), device_to_host_r),
    }
  );

  return programs;
}

int main (int argc, char** argv) {

  try {
    // Get options from command line arguments / defaults. (see utils.hpp)
    auto options = utils::parseOptions(argc, argv);
    testUpperBoundDt(options);

    // Attach to IPU device
    auto device = getDevice(options.num_ipus);
    auto &target = device.getTarget();
    options.architecture = target.getTargetArchString();
    options.num_tiles_available = target.getNumTiles();
    workDivision(options); // requires options.num_tiles_availble
    printGeneralInfo(options);

    // Host variables
    unsigned area = options.height*options.width;
    unsigned inner_area = (options.height - 2)*(options.width - 2);
    std::vector<float> initial_e(area);
    std::vector<float> initial_r(area);
    std::vector<float> ipu_results_e(area);
    std::vector<float> ipu_results_r(area);

    // Initial values: 
    // e: left half=0, right half=1
    // r: bottom half=0, top half=1
    for (std::size_t x = 0; x < options.width; ++x) {
      for (std::size_t y = 0; y < options.height; ++y) {
        if (x < options.height/2) {
          initial_r[y + x*options.width] = 1.0;
        } else {
          initial_r[y + x*options.width] = 0.0;
        }
        if (y < options.width/2) {
          initial_e[y + x*options.width] = 0.0;
        } else {
          initial_e[y + x*options.width] = 1.0;
        }
      }
    }

    // // Create graph object
    // poplar::Graph graph{target};
    // graph.addCodelets("codelets.gp");

    // // Create programs
    // auto programs = createIpuPrograms(graph, options);
    
    // // Compile graph and programs
    // auto exe = poplar::compileGraph(graph, programs);
    
    // // Create Engine object
    // poplar::Engine engine(std::move(exe));
    // engine.connectStream("host_to_device_stream_e", &initial_e[0], &initial_e[area]);
    // engine.connectStream("host_to_device_stream_r", &initial_r[0], &initial_r[area]);
    // engine.connectStream("device_to_host_stream_e", &ipu_results_e[0], &ipu_results_e[area]);
    // engine.connectStream("device_to_host_stream_r", &ipu_results_r[0], &ipu_results_r[area]);
    // engine.load(device);
    // engine.run(0);
    // auto start = std::chrono::steady_clock::now();
    // engine.run(1); // Compute set execution
    // auto stop = std::chrono::steady_clock::now();
    // engine.run(2);

    // // Results
    // auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    // double wall_time = 1e-9*diff.count();
    // double flops_per_element = 27.0;
    // double flops = inner_area * options.num_iterations * flops_per_element / wall_time;
    // double loaded_elems_per_stencil = 6 + 2; // 6 for e, 2 for r
    // double stored_elems_per_stencil = 2; // 1 for e, 1 for r
    // double total_elems_per_stencil = loaded_elems_per_stencil + stored_elems_per_stencil;
    // double bandwidth_base = inner_area * options.num_iterations * sizeof(float) / wall_time;
    // double load_bw = loaded_elems_per_stencil*bandwidth_base;
    // double store_bw = stored_elems_per_stencil*bandwidth_base;
    // double total_bw = total_elems_per_stencil*bandwidth_base;
    // printPerformance(flops, load_bw, store_bw, total_bw);

    if (options.cpu) {
      std::vector<float> cpu_results_e(area);
      std::vector<float> cpu_results_r(area);
      solveAlievPanfilovCpu(initial_e, initial_r, cpu_results_e, cpu_results_r, options);
      // reportCpuVsIpu(cpu_results_e, cpu_results_r, ipu_results_e, ipu_results_r, options);
    }

    // End of try block
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}