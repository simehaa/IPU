#include <chrono>
#include <math.h>
#include <cmath>
#include <iostream>
#include <iomanip>

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>

#include "utils.hpp"

/*
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
   * Compute Set which performs heat equation once from 
   * tensor "in" to tensor "out"
   */
  auto compute_set = graph.addComputeSet(compute_set_name);
  unsigned num_workers_per_tile = graph.getTarget().getNumWorkerContexts();
  unsigned nh = options.splits[0]; // Number of splits in height
  unsigned nw = options.splits[1]; // Number of splits in width
  unsigned nd = options.splits[2]; // Number of splits in depth
  unsigned nwh = 2; // Splits in height (among workers on a tile)
  unsigned nww = 3; // Splits in width (among workers on a tile)
  std::size_t halo_volume = 0;

  for (std::size_t ipu = 0; ipu < options.num_ipus; ++ipu) {

    // Ensure overlapping grids among the IPUs
    std::size_t offset_back = 2;
    auto ipu_in_slice = in.slice(
      {0, 0, block_low(ipu, options.num_ipus, options.depth-2)},
      {options.height, options.width, block_high(ipu, options.num_ipus, options.depth-2) + offset_back}
    );
    auto ipu_out_slice = out.slice(
      {0, 0, block_low(ipu, options.num_ipus, options.depth-2)},
      {options.height, options.width, block_high(ipu, options.num_ipus, options.depth-2) + offset_back}
    );
    std::size_t inter_depth = ipu_in_slice.shape()[2];

    for (std::size_t x = 0; x < nh; ++x) {
      for (std::size_t y = 0; y < nw; ++y) {
        for (std::size_t z = 0; z < nd; ++z) {
          unsigned tile_id = index(x, y, z, nw, nd) + ipu*options.tiles_per_ipu;
          unsigned tile_x = block_low(x, nh, options.height-2) + 1;
          unsigned tile_y = block_low(y, nw, options.width-2) + 1;
          unsigned tile_height = block_size(x, nh, options.height-2);
          unsigned tile_width = block_size(y, nw, options.width-2);
          unsigned z_low = block_low(z, nd, inter_depth-2) + 1;
          unsigned z_high = block_high(z, nd, inter_depth-2) + 1;
          unsigned tile_depth = z_high - z_low;

          std::vector<std::size_t> shape = {tile_height, tile_width, tile_depth};
          if (volume(shape) < volume(options.smallest_slice))
            options.smallest_slice = shape;
          if (volume(shape) > volume(options.largest_slice)) 
            options.largest_slice = shape;

          if ((x == 0) || (x == nh - 1)) {
            halo_volume += tile_width*tile_depth;
          } else {
            halo_volume += 2*tile_width*tile_depth;
          }
          if ((y == 0) || (y == nw - 1)) { 
            halo_volume += tile_height*tile_depth;
          } else {
            halo_volume += 2*tile_height*tile_depth;
          }
          if ((z == 0) || (z == nd - 1)) { 
            halo_volume += tile_height*tile_width;
          } else {
            halo_volume += 2*tile_height*tile_width;
          }

          for (std::size_t worker_xi = 0; worker_xi < nwh; ++worker_xi) {
            for (std::size_t worker_yi = 0; worker_yi < nww; ++worker_yi) {
              
              // Dividing tile work among workers
              unsigned x_low = tile_x + block_low(worker_xi, nwh, tile_height);
              unsigned x_high = tile_x + block_high(worker_xi, nwh, tile_height);
              unsigned y_low = tile_y + block_low(worker_yi, nww, tile_width);
              unsigned y_high = tile_y + block_high(worker_yi, nww, tile_width);

              // NOTE: include overlap for "in_slice"
              auto in_slice = ipu_in_slice.slice(
                {x_low-1, y_low-1, z_low-1},
                {x_high+1, y_high+1, z_high+1}
              );

              auto out_slice = ipu_out_slice.slice(
                {x_low, y_low, z_low},
                {x_high, y_high, z_high}
              );

              // Assign vertex to graph
              auto v = graph.addVertex(compute_set, options.vertex);
              graph.connect(v["in"], in_slice.flatten(0,2));
              graph.connect(v["out"], out_slice.flatten(0,2));
              graph.setInitialValue(v["worker_height"], x_high - x_low);
              graph.setInitialValue(v["worker_width"], y_high - y_low);
              graph.setInitialValue(v["worker_depth"], z_high - z_low);
              graph.setInitialValue(v["alpha"], options.alpha);
              graph.setTileMapping(v, tile_id);
            }
          }
        }
      }
    }
  }
  options.halo_volume = halo_volume;

  return compute_set;
}

std::vector<poplar::program::Program> createIpuPrograms(
  poplar::Graph &graph, utils::Options &options) { 

  // Allocate Tensors, device variables
  auto a = graph.addVariable(poplar::FLOAT, {options.height, options.width, options.depth}, "a");
  auto b = graph.addVariable(poplar::FLOAT, {options.height, options.width, options.depth}, "b");
  
  for (std::size_t ipu = 0; ipu < options.num_ipus; ++ipu) {

    // Partition the entire grid FIRST among IPUs by splitting in depth (dimension 2)
    std::size_t offset_front = (ipu == 0) ? 0 : 1;
    std::size_t offset_back = (ipu == options.num_ipus - 1) ? 2 : 1;
    auto ipu_slice = a.slice(
      {
        0, 
        0, 
        block_low(ipu, options.num_ipus, options.depth-2) + offset_front
      },
      {
        options.height, 
        options.width, 
        block_high(ipu, options.num_ipus, options.depth-2) + offset_back
      }
    );

    // Evaluate depth of THIS slice
    std::size_t inter_depth = ipu_slice.shape()[2];

    // Next, tile map 1472 slices per IPU
    for (std::size_t tile_x = 0; tile_x < options.splits[0]; ++tile_x) {
      for (std::size_t tile_y = 0; tile_y < options.splits[1]; ++tile_y) {
        for (std::size_t tile_z = 0; tile_z < options.splits[2]; ++tile_z) {

          unsigned tile_id = index(tile_x, tile_y, tile_z, options.splits[1], options.splits[2]) + ipu*options.tiles_per_ipu;

          // Evaluate offsets in all dimensions (avoid overlap at edges)
          std::size_t offset_top = (tile_x == 0) ? 0 : 1;
          std::size_t offset_left = (tile_y == 0) ? 0 : 1;
          std::size_t inter_offset_front = (tile_z == 0) ? 0 : 1;
          std::size_t offset_bottom = (tile_x == options.splits[0] - 1) ? 2 : 1;
          std::size_t offset_right = (tile_y == options.splits[1] - 1) ? 2 : 1;
          std::size_t inter_offset_back = (tile_z == options.splits[2] - 1) ? 2 : 1;

          auto tile_slice = ipu_slice.slice(
            {
              block_low(tile_x, options.splits[0], options.height-2) + offset_top, 
              block_low(tile_y, options.splits[1], options.width-2) + offset_left,
              block_low(tile_z, options.splits[2], inter_depth-2) + inter_offset_front
            },
            {
              block_high(tile_x, options.splits[0], options.height-2) + offset_bottom, 
              block_high(tile_y, options.splits[1], options.width-2) + offset_right,
              block_high(tile_z, options.splits[2], inter_depth-2) + inter_offset_back
            }
          );
          
          graph.setTileMapping(tile_slice, tile_id);
        }
      }
    }
  }

  // Apply the tile mapping of "a" to be the same for "b"
  const auto& tile_mapping = graph.getTileMapping(a);
  graph.setTileMapping(b, tile_mapping);

  // Define data streams
  std::size_t volume = options.height*options.width*options.depth;
  auto host_to_device = graph.addHostToDeviceFIFO("host_to_device_stream", poplar::FLOAT, volume);
  auto device_to_host = graph.addDeviceToHostFIFO("device_to_host_stream", poplar::FLOAT, volume);

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
  try {
    // Get options from command line arguments / defaults. (see utils.hpp)
    auto options = utils::parseOptions(argc, argv);

    // Set up of 3D mesh properties
    std::size_t base_length = 320;
    options.side = side_length(options.num_ipus, base_length);
    // printMultiIpuGridInfo(base_length);
    
    // Check if user wanted to override dimensions (only do it if all three were given)
    if (options.height == 0 && options.width == 0 && options.depth == 0) {
      // This block: none of the three were given, therefore
      // construct a cubic mesh (default)
      options.height = options.side;
      options.width = options.side;
      options.depth = options.side;
    } else if (options.height != 0 && options.width != 0 && options.depth != 0) {
      // This block: all three were given, hence options.height, options.width, and
      // options.depth are set and good to go
    } else {
      // This block, means that one or two (but not all three) were given
      // Since that is ambiguous, use the default cubic mesh instead, 
      // and warn the user to either provide ALL three or NONE
      std::cout 
        << "Warning: to override dimensions, please provide all three: "
        << "--height <h> --width <w> --depth <d>. Using " << options.side 
        << " for all dimensions instead.\n";
      options.height = options.side;
      options.width = options.side;
      options.depth = options.side;
    }

    // Attach to IPU device
    auto device = getDevice(options);
    auto &target = device.getTarget();
    options.num_tiles_available = target.getNumTiles();
    options.tiles_per_ipu = options.num_tiles_available / options.num_ipus;
    workDivision(options);

    // Create graph object
    poplar::Graph graph{target};
    graph.addCodelets("codelets.gp");
    
    std::size_t inner_volume = (options.height - 2) * (options.width - 2) * (options.depth - 2);
    std::size_t total_volume = (options.height * options.width * options.depth);
    std::vector<float> ipu_results(total_volume); 
    std::vector<float> initial_values(total_volume);
    std::vector<float> cpu_results(total_volume);

    // initialize initial values with random floats
    for (std::size_t i = 0; i < total_volume; ++i)
      initial_values[i] = randomFloat();

    // perform CPU execution (and later compute MSE in IPU vs. CPU execution)
    if (options.cpu) 
      cpu_results = heatEquationCpu(initial_values, options);
    
    // Setup of programs, graph and engine
    auto programs = createIpuPrograms(graph, options);
    auto exe = poplar::compileGraph(graph, programs);
    poplar::Engine engine(std::move(exe));
    engine.connectStream("host_to_device_stream", &initial_values[0], &initial_values[total_volume]);
    engine.connectStream("device_to_host_stream", &ipu_results[0], &ipu_results[total_volume]);
    engine.load(device);

    std::size_t num_program_steps = programs.size();
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

    // End of try block
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
  return 0;
}

