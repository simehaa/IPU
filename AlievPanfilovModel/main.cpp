#include <chrono>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include "utils.hpp"

poplar::ComputeSet createComputeSet(
  poplar::Graph &graph,
  poplar::Tensor &e_in,
  poplar::Tensor &e_out,
  poplar::Tensor &r,
  utils::Options &options,
  const std::string& compute_set_name,
  const std::string& vertex) {

  auto compute_set = graph.addComputeSet(compute_set_name);
  unsigned num_workers_per_tile = graph.getTarget().getNumWorkerContexts();
  unsigned nh = options.nh; // Number of splits in height
  unsigned nw = options.nw; // Number of splits in width
  std::size_t halo_volume = 0;

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
        auto e_in_slice = e_in.slice({x_low-1, y_low-1}, {x_high+1, y_high+1}); // padding of 1 compared to tile mapping
        auto e_out_slice = e_out.slice({x_low, y_low}, {x_high, y_high}); // correspond exactly to tile mapping
        auto r_slice = r.slice({x_low-1, y_low-1}, {x_high-1, y_high-1}); // correspond exactly to tile mapping

        std::vector<std::size_t> shape = e_out_slice.shape();
        if (area(shape) < area(options.smallest_slice))
          options.smallest_slice = shape;
        if (area(shape) > area(options.largest_slice)) 
          options.largest_slice = shape;

        if ((x == 0) || (x == nh - 1)) {
          halo_volume += tile_width;
        } else {
          halo_volume += 2*tile_width;
        }
        if ((y == 0) || (y == nw - 1)) { 
          halo_volume += tile_height;
        } else {
          halo_volume += 2*tile_height;
        }

        // Assign vertex to graph
        auto v = graph.addVertex(compute_set, vertex);
        graph.connect(v["e_in"], e_in_slice);
        graph.connect(v["e_out"], e_out_slice);
        graph.connect(v["r"], r_slice);
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
  options.halo_volume = halo_volume;

  return compute_set;
}

std::vector<poplar::program::Program> createIpuPrograms(
  poplar::Graph &graph,
  utils::Options &options) { 

  // Allocate Tensors, device variables
  auto e_a = graph.addVariable(poplar::FLOAT, {options.height + 2, options.width + 2}, "e_a");
  auto e_b = graph.addVariable(poplar::FLOAT, {options.height + 2, options.width + 2}, "e_b");
  auto r = graph.addVariable(poplar::FLOAT, {options.height, options.width}, "r");

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
      std::size_t x_low = block_low(x, nh, options.height);
      std::size_t y_low = block_low(y, nw, options.width);
      std::size_t x_high = block_high(x, nh, options.height);
      std::size_t y_high = block_high(y, nw, options.width);

      // map a slice to a tile
      graph.setTileMapping(
        e_a.slice(
          {x_low + offset_top, y_low + offset_left},
          {x_high + offset_bottom, y_high + offset_right}
        ),
        tile_id
      );

      graph.setTileMapping(
        r.slice(
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

  // Define data streams
  long unsigned area = options.height*options.width;
  auto host_to_device_e = graph.addHostToDeviceFIFO("host_to_device_stream_e", poplar::FLOAT, area);
  auto host_to_device_r = graph.addHostToDeviceFIFO("host_to_device_stream_r", poplar::FLOAT, area);
  auto device_to_host_e = graph.addDeviceToHostFIFO("device_to_host_stream_e", poplar::FLOAT, area);
  auto device_to_host_r = graph.addDeviceToHostFIFO("device_to_host_stream_r", poplar::FLOAT, area);

  std::vector<poplar::program::Program> programs;

  // Program 0: move content of initial_values into all four tensors
  long unsigned h = options.height + 2;
  long unsigned w = options.width + 2;
  programs.push_back(
    poplar::program::Sequence{
      poplar::program::Copy(host_to_device_e, e_a.slice({1,1},{h-1,w-1})),
      poplar::program::Copy(host_to_device_r, r),
      poplar::program::Copy(e_a, e_b),
    }
  );

  // Program 1: execution(s) of compute sets
  poplar::program::Sequence execute_this_compute_set;
  auto compute_set_b_to_a = createComputeSet(graph, e_b, e_a, r, options, "Aliev_Panfilov_b_to_a", "AlievPanfilov");
  auto compute_set_a_to_b = createComputeSet(graph, e_a, e_b, r, options, "Aliev_Panfilov_a_to_b", "AlievPanfilov");
  
  auto a_to_b = poplar::program::Sequence({
    poplar::program::Copy(e_a.slice({2,1},{3,w-1}), e_a.slice({0,1},{1,w-1})), // north
    poplar::program::Copy(e_a.slice({h-3,1},{h-2,w-1}), e_a.slice({h-1,1},{h,w-1})), // south
    poplar::program::Copy(e_a.slice({1,2},{h-1,3}), e_a.slice({1,0},{h-1,1})), // west
    poplar::program::Copy(e_a.slice({1,w-3},{h-1,w-2}), e_a.slice({1,w-1},{h-1,w})), // east
    poplar::program::Execute(compute_set_a_to_b)
  });

  auto b_to_a = poplar::program::Sequence({
    poplar::program::Copy(e_b.slice({2,1},{3,w-1}), e_b.slice({0,1},{1,w-1})), // north
    poplar::program::Copy(e_b.slice({h-3,1},{h-2,w-1}), e_b.slice({h-1,1},{h,w-1})), // south
    poplar::program::Copy(e_b.slice({1,2},{h-1,3}), e_b.slice({1,0},{h-1,1})), // west
    poplar::program::Copy(e_b.slice({1,w-3},{h-1,w-2}), e_b.slice({1,w-1},{h-1,w})), // east
    poplar::program::Execute(compute_set_b_to_a)
  });

  if (options.num_iterations % 2 == 1) // if num_iterations is odd: add one extra iteration
    execute_this_compute_set.add(a_to_b);

  // add iterations 
  execute_this_compute_set.add(
    poplar::program::Repeat(
      options.num_iterations/2,
      poplar::program::Sequence({b_to_a, a_to_b})
    )
  );
  programs.push_back(execute_this_compute_set);

  // Program 2: copy final results back to host
  programs.push_back(
    poplar::program::Sequence({
      poplar::program::Copy(e_b.slice({1,1},{h-1,w-1}), device_to_host_e),
      poplar::program::Copy(r, device_to_host_r),
    })
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
    std::vector<float> initial_e(area);
    std::vector<float> initial_r(area);
    std::vector<float> ipu_results_e(area);
    std::vector<float> ipu_results_r(area);

    // Initial values: 
    // e: left half=0, right half=1
    // r: bottom half=0, top half=1
    for (std::size_t x = 0; x < options.height; ++x) {
      for (std::size_t y = 0; y < options.width; ++y) {
        initial_r[y + x*options.width] = (x < options.height/2) ? 1.0 : 0.0;
        initial_e[y + x*options.width] = (y < options.width/2) ? 0.0 : 1.0;
      }
    }

    // // Create graph object
    poplar::Graph graph{target};
    graph.addCodelets("codelets.gp");
    auto programs = createIpuPrograms(graph, options);
    auto exe = poplar::compileGraph(graph, programs);
    
    // Create Engine object
    poplar::Engine engine(std::move(exe));
    engine.connectStream("host_to_device_stream_e", &initial_e[0], &initial_e[area]);
    engine.connectStream("host_to_device_stream_r", &initial_r[0], &initial_r[area]);
    engine.connectStream("device_to_host_stream_e", &ipu_results_e[0], &ipu_results_e[area]);
    engine.connectStream("device_to_host_stream_r", &ipu_results_r[0], &ipu_results_r[area]);
    engine.load(device);
    engine.run(0);
    auto start = std::chrono::steady_clock::now();
    engine.run(1); // Compute set execution
    auto stop = std::chrono::steady_clock::now();
    engine.run(2);

    // Results
    auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    double wall_time = 1e-9*diff.count();
    printPerformance(wall_time, options);

    if (options.cpu) {
      std::vector<float> cpu_results_e(area);
      std::vector<float> cpu_results_r(area);
      solveAlievPanfilovCpu(initial_e, initial_r, cpu_results_e, cpu_results_r, options);
      reportCpuVsIpu(cpu_results_e, cpu_results_r, ipu_results_e, ipu_results_r, options);
    }

    // End of try block
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}