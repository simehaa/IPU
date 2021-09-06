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
  const std::string& compute_set_name,
  const std::string& vertex) {
  /*
   * Compute Set which performs the isotropic diffusion 
   * (one iteration of sliding the stencil over the inner elements)
   * once from tensor "in" to tensor "out"
   */
  auto compute_set = graph.addComputeSet(compute_set_name);
  unsigned num_workers_per_tile = graph.getTarget().getNumWorkerContexts();
  unsigned nh = options.nh; // Number of splits in height
  unsigned nw = options.nw; // Number of splits in width

  // this double loop essentially iterated over the number of tiles (see tile_id below)
  for (std::size_t x = 0; x < nh; ++x) {
    for (std::size_t y = 0; y < nw; ++y) {

      // tile_id should run from 0, 1, ..., nh*nw
      unsigned tile_id = index(x, y, nw);

      // Start indices
      unsigned tile_x = block_low(x, nh, options.height-2);
      unsigned tile_y = block_low(y, nw, options.width-2);

      // Tile sub-grid sizes which include overlap
      unsigned tile_height = block_size(x, nh, options.height-2);
      unsigned tile_width = block_size(y, nw, options.width-2);
      unsigned tile_area = tile_height * tile_width;

      // Finding the largest subdivision in each respective dimension
      if (tile_area > options.largest_tile_area) {
        options.largest_tile_height = tile_height;
        options.largest_tile_width = tile_width;
        options.largest_tile_area = tile_area;
      }

      // Finding the smallest subdivision in each respective dimension
      if (tile_area < options.smallest_tile_area) {
        options.smallest_tile_height = tile_height;
        options.smallest_tile_width = tile_width;
        options.smallest_tile_area = tile_area;
      }

      if (tile_area > 0 && options.first_compute_set) // ONLY count this once (hence the first_compute_set check)
        options.num_tiles++;        

      for (std::size_t worker_i = 0; worker_i < num_workers_per_tile; ++worker_i) {
        
        // Dividing tile work among workers by splitting up grid further in height (x) dimension
        unsigned x_low = tile_x + block_low(worker_i, num_workers_per_tile, tile_height) + 1;
        unsigned worker_height = block_size(worker_i, num_workers_per_tile, tile_height);
        unsigned x_high = x_low + worker_height;
        
        unsigned y_low = tile_y + 1;
        unsigned worker_width = tile_width;
        unsigned y_high = y_low + worker_width;

        // Assign vertex to graph
        auto v = graph.addVertex(compute_set, vertex);
        graph.connect(v["in"], in.slice({x_low-1, y_low-1}, {x_high+1, y_high+1}));
        graph.connect(v["out"], out.slice({x_low, y_low}, {x_high, y_high}));
        graph.setInitialValue(v["worker_height"], worker_height);
        graph.setInitialValue(v["worker_width"], worker_width);
        graph.setInitialValue(v["alpha"], options.alpha);
        graph.setTileMapping(v, options.random_perm[tile_id]);
      }
    }
  }
  options.first_compute_set = false; // Compute set is used to e.g. keep track of sub-grid
  // area and side-lengths, but thi is only done when first_compute_set = true.

  return compute_set;
}

std::vector<poplar::program::Program> createIpuPrograms(
  poplar::Graph &graph,
  std::vector<float> initial_values,
  utils::Options &options) { 
  /*
   * This function defines a vector of IPU programs which are set to execute
   * the diffusion equation (PDE) on a three dimensional grid.
   *
   * The returned vector consist of three Poplar programs:
   * 0: Stream the 3D grid to device - both graph tensors "a" and "b" will get these values
   * 1: Execute diffusion eq. back and forth between "a" and "b"
   * 2: Stream the results back from device (will always be "b")
   */

  // Allocate Tensors, device variables
  auto a = graph.addVariable(poplar::FLOAT, {options.height, options.width}, "a");
  auto b = graph.addVariable(poplar::FLOAT, {options.height, options.width}, "b");

  // Tile mappings
  int nh = options.nh; // Number of splits in height
  int nw = options.nw; // Number of splits in width

  for (std::size_t x = 0; x < nh; ++x) {
    for (std::size_t y = 0; y < nw; ++y) {

      // tile_id runs from 0, 1, 2, ..., nh*nw
      unsigned tile_id = index(x, y, nw);

      // Evaluate offsets in all dimensions (avoid overlap at edges)
      int offset_top = (x == 0) ? 0 : 1;
      int offset_left = (y == 0) ? 0 : 1;
      int offset_bottom = (x == nh - 1) ? 2 : 1;
      int offset_right = (y == nw - 1) ? 2 : 1;

      // map a slice to a tile
      graph.setTileMapping(
        a.slice(
          {
            block_low(x, nh, options.height-2) + offset_top, 
            block_low(y, nw, options.width-2) + offset_left
          },
          {
            block_high(x, nh, options.height-2) + offset_bottom, 
            block_high(y, nw, options.width-2) + offset_right
          }
        ),
        options.random_perm[tile_id]
      );
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

  auto vertices = utils::getVertices();
  for (auto vertex : vertices) {
    // Create compute sets
    auto compute_set_b_to_a = createComputeSet(graph, b, a, options, "HeatEquation_b_to_a", vertex);
    auto compute_set_a_to_b = createComputeSet(graph, a, b, options, "HeatEquation_a_to_b", vertex);
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
  }

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
    options.architecture = target.getTargetArchString();
    options.num_tiles_available = target.getNumTiles();
    options.memory_available = target.getMemoryBytes();
    for (std::size_t i = 0; i < options.num_tiles_available; ++i) {
      options.random_perm.push_back(i);
    }
    if (options.random_tiles) {
      std::random_shuffle(options.random_perm.begin(), options.random_perm.end());
    }
    workDivision(options);

    // Create graph object
    poplar::Graph graph{target};
    graph.addCodelets("codelets.gp");

    // Host variables (Load image or generate 2D grid)
    std::vector<float> initial_values = initializeValues(options);
    unsigned area = initial_values.size();
    double inner_area = (options.height - 2.0) * (options.width - 2.0);
    double total_area = options.height * options.width;
    double flops_per_element = 6.0; // mid*gamma + alpha*(top + bottom + left + right)
    std::vector<float> ipu_results(area); // empty grid that can obtain the IPU result
    std::vector<float> cpu_results(area);
    std::vector<double> TFLOPS;
    std::vector<double> bandwidth_TB_S;
    std::vector<double> MSE;
    auto vertices = utils::getVertices();
    if (options.cpu) // perform CPU execution (and later compute MSE in IPU vs. CPU execution)
      cpu_results = heatEquationCpu(initial_values, options);

    auto programs = createIpuPrograms(graph, initial_values, options);
    auto exe = poplar::compileGraph(graph, programs);
    poplar::Engine engine(std::move(exe));
    engine.connectStream("host_to_device_stream", &initial_values[0], &initial_values[area]);
    engine.connectStream("device_to_host_stream", &ipu_results[0], &ipu_results[area]);
    engine.load(device);

    int num_program_steps = programs.size();
    engine.run(0); // stream data to device

    for (std::size_t i = 1; i < num_program_steps; i+= 2) {
      auto start = std::chrono::steady_clock::now();
      engine.run(i); // Compute set execution
      auto stop = std::chrono::steady_clock::now();
      engine.run(i+1); // Stream of results

      // Report
      auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
      double wall_time = 1e-9*diff.count();
      double flops = inner_area * options.num_iterations * flops_per_element / wall_time;
      double bandwidth = inner_area * options.num_iterations * sizeof(float) / wall_time;
      TFLOPS.push_back(flops*1e-12);
      bandwidth_TB_S.push_back(bandwidth*1e-12);

      if (options.cpu)
        MSE.push_back(meanSquaredErrorInnerElements(ipu_results, cpu_results, options));
    }

    printGeneralInfo(options);
    printLatexTabular(vertices, TFLOPS, bandwidth_TB_S);

    if (options.cpu)
      printMeanSquaredError(vertices, MSE);

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