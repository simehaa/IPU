#include <chrono>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include "utils.hpp"

/*
Conventions
-----------
Functions: camelCase
Variables: underscore_separated_words
*/

poplar::ComputeSet createComputeSet(
  poplar::Graph &graph,
  poplar::Tensor &in,
  poplar::Tensor &out,
  utils::Options &options,
  const std::string& compute_set_name) {
  /*
   * Compute Set which performs the isotropic diffusion 
   * (one iteration of sliding the stencil over the inner elements)
   * once from tensor "in" to tensor "out"
   */
  auto compute_set = graph.addComputeSet(compute_set_name);
  unsigned num_workers_per_tile = graph.getTarget().getNumWorkerContexts();
  unsigned nh = options.splits[0]; // Number of splits in height
  unsigned nw = options.splits[1]; // Number of splits in width
  std::size_t halo_volume = 0;

  // this double loop essentially iterated over the number of tiles (see tile_id below)
  for (std::size_t ipu = 0; ipu < options.num_ipus; ++ipu) {

     // Ensure overlapping grids among the IPUs
    std::size_t offset_right = 2;
    auto ipu_in_slice = in.slice(
      {0, block_low(ipu, options.num_ipus, options.width-2)},
      {options.height, block_high(ipu, options.num_ipus, options.width-2) + offset_right}
    );
    auto ipu_out_slice = out.slice(
      {0, block_low(ipu, options.num_ipus, options.width-2)},
      {options.height, block_high(ipu, options.num_ipus, options.width-2) + offset_right}
    );
    std::size_t inter_width = ipu_in_slice.shape()[1];

    for (std::size_t x = 0; x < nh; ++x) {
      for (std::size_t y = 0; y < nw; ++y) {
        unsigned tile_id = index(x, y, nw) + ipu*options.tiles_per_ipu;
        unsigned tile_x = block_low(x, nh, options.height-2) + 1;
        unsigned tile_height = block_size(x, nh, options.height-2);
        unsigned y_low = block_low(y, nw, inter_width-2) + 1;
        unsigned y_high = block_high(y, nw, inter_width-2) + 1;
        unsigned tile_width = y_high - y_low;

        std::vector<std::size_t> shape = {tile_height, tile_width};
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

        for (std::size_t worker_i = 0; worker_i < num_workers_per_tile; ++worker_i) {
          
          // Dividing tile work among workers by splitting up grid further in height (x) dimension
          unsigned x_low = tile_x + block_low(worker_i, num_workers_per_tile, tile_height);
          unsigned x_high = tile_x + block_high(worker_i, num_workers_per_tile, tile_height);
          
          // NOTE: include overlap for "in_slice"
          auto in_slice = ipu_in_slice.slice(
            {x_low-1, y_low-1},
            {x_high+1, y_high+1}
          );

          auto out_slice = ipu_out_slice.slice(
            {x_low, y_low},
            {x_high, y_high}
          );

          // Assign vertex to graph
          auto v = graph.addVertex(compute_set, options.vertex);
          graph.connect(v["in"], in_slice);
          graph.connect(v["out"], out_slice);
          graph.setInitialValue(v["worker_height"], x_high - x_low);
          graph.setInitialValue(v["worker_width"], y_high - y_low);
          graph.setInitialValue(v["alpha"], options.alpha);
          graph.setTileMapping(v, tile_id);
        }
      }
    }
  }
  options.halo_volume = halo_volume;

  return compute_set;
}

std::vector<poplar::program::Program> createIpuPrograms(
  poplar::Graph &graph,
  std::vector<float> initial_values,
  utils::Options &options) { 
  /*
   * This function defines a vector of IPU programs which are set to execute
   * the diffusion equation (PDE) on a two dimensional grid.
   *
   * The returned vector consist of three Poplar programs:
   * 0: Stream the 2D grid to device - both graph tensors "a" and "b" will get these values
   * 1: Execute diffusion eq. back and forth between "a" and "b"
   * 2: Stream the results back from device (will always be "b")
   */

  // Allocate Tensors, device variables
  auto a = graph.addVariable(poplar::FLOAT, {options.height, options.width}, "a");
  auto b = graph.addVariable(poplar::FLOAT, {options.height, options.width}, "b");

  // Tile mappings
  for (std::size_t ipu = 0; ipu < options.num_ipus; ++ipu) {

    std::size_t offset_left = (ipu == 0) ? 0 : 1;
    std::size_t offset_right = (ipu == options.num_ipus - 1) ? 2 : 1;
    auto ipu_slice = a.slice(
      {
        0, 
        block_low(ipu, options.num_ipus, options.width-2) + offset_left
      },
      {
        options.height, 
        block_high(ipu, options.num_ipus, options.width-2) + offset_right
      }
    );
    std::size_t inter_width = ipu_slice.shape()[1];

    for (std::size_t tile_x = 0; tile_x < options.splits[0]; ++tile_x) {
      for (std::size_t tile_y = 0; tile_y < options.splits[1]; ++tile_y) {

        // tile_id runs from 0, 1, 2, ..., nh*nw
        unsigned tile_id = index(tile_x, tile_y, options.splits[1]) + ipu*options.tiles_per_ipu;

        // Evaluate offsets in all dimensions (avoid overlap at edges)
        int offset_top = (tile_x == 0) ? 0 : 1;
        int inter_offset_left = (tile_y == 0) ? 0 : 1;
        int offset_bottom = (tile_x == options.splits[0] - 1) ? 2 : 1;
        int inter_offset_right = (tile_y == options.splits[1] - 1) ? 2 : 1;

        // map a slice to a tile
        auto tile_slice = ipu_slice.slice(
          {
            block_low(tile_x, options.splits[0], options.height-2) + offset_top, 
            block_low(tile_y, options.splits[1], inter_width-2) + inter_offset_left
          },
          {
            block_high(tile_x, options.splits[0], options.height-2) + offset_bottom, 
            block_high(tile_y, options.splits[1], inter_width-2) + inter_offset_right
          }
        );

        graph.setTileMapping(tile_slice, tile_id);
      }
    }
  }
  
  // Apply the tile mapping of "a" to be the same for "b"
  const auto& tile_mapping = graph.getTileMapping(a);
  graph.setTileMapping(b, tile_mapping);

  // Define data streams
  long unsigned area = options.height*options.width;
  auto host_to_device = graph.addHostToDeviceFIFO("host_to_device_stream", poplar::FLOAT, area);
  auto device_to_host = graph.addDeviceToHostFIFO("device_to_host_stream", poplar::FLOAT, area);

  std::vector<poplar::program::Program> programs;

  // Program 0: move content of initial_values into both device variables a and b
  programs.push_back(
    poplar::program::Sequence{
      poplar::program::Copy(host_to_device, b), // initial_values to b
      poplar::program::Copy(b, a), // initial_values to a
    }
  );

  // Create compute sets
  auto compute_set_b_to_a = createComputeSet(graph, b, a, options, "HeatEquation_b_to_a");
  auto compute_set_a_to_b = createComputeSet(graph, a, b, options, "HeatEquation_a_to_b");
  poplar::program::Sequence execute_this_compute_set;

  if (options.num_iterations % 2 == 1) { // if num_iterations is odd: add one extra iteration
    execute_this_compute_set.add(poplar::program::Execute(compute_set_a_to_b));
  }

  // add iterations 
  execute_this_compute_set.add(
    poplar::program::Repeat(
      options.num_iterations/2,
      poplar::program::Sequence{
        poplar::program::Execute(compute_set_b_to_a),
        poplar::program::Execute(compute_set_a_to_b)
      }
    )
  );

  programs.push_back(execute_this_compute_set);
  programs.push_back(poplar::program::Copy(b, device_to_host));

  return programs;
}

int main (int argc, char** argv) {
  GOOGLE_PROTOBUF_VERIFY_VERSION; // for image

  try {
    // Get options from command line arguments / defaults. (see utils.hpp)
    auto options = utils::parseOptions(argc, argv);

    // Attach to IPU device
    auto device = getDevice(options.num_ipus);
    auto &target = device.getTarget();
    options.num_tiles_available = target.getNumTiles();
    options.tiles_per_ipu = options.num_tiles_available / options.num_ipus;
    std::vector<float> initial_values = initializeValues(options);
    workDivision(options); // AFTER initalizeValues()

    // Create graph object
    poplar::Graph graph{target};
    graph.addCodelets("codelets.gp");

    // Host variables (Load image or generate 2D grid)
    unsigned area = initial_values.size();
    double inner_area = (options.height - 2.0) * (options.width - 2.0);
    double total_area = options.height * options.width;
    double flops_per_element = 6.0; // mid*gamma + alpha*(top + bottom + left + right)
    std::vector<float> ipu_results(area); // empty grid that can obtain the IPU result
    std::vector<float> cpu_results(area);

    if (options.cpu) // perform CPU execution (and later compute MSE in IPU vs. CPU execution)
      cpu_results = heatEquationCpu(initial_values, options);

    auto programs = createIpuPrograms(graph, initial_values, options);
    auto exe = poplar::compileGraph(graph, programs);
    poplar::Engine engine(std::move(exe));
    engine.connectStream("host_to_device_stream", &initial_values[0], &initial_values[total_area]);
    engine.connectStream("device_to_host_stream", &ipu_results[0], &ipu_results[total_area]);
    engine.load(device);

    engine.run(0); // stream data to device
    auto start = std::chrono::steady_clock::now();
    engine.run(1); // Compute set execution
    auto stop = std::chrono::steady_clock::now();
    engine.run(2); // Stream of results

    // Report
    auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    double wall_time = 1e-9*diff.count();
    printResults(options, wall_time);

    if (options.cpu)
      printMeanSquaredError(ipu_results, cpu_results, options);

    if (options.in_file.length() != 0)
      saveImage(options, ipu_results);

    // End of try block
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
  google::protobuf::ShutdownProtobufLibrary();

  return EXIT_SUCCESS;
}