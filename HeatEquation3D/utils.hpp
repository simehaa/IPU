#pragma once
#include <boost/program_options.hpp>
#include <string>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <bits/stdc++.h>

/*
README
To avoid confusion in indexing:
  x goes along heigth
  y goes along width
  z goes along depth

All triple nested loops go:
for x in height
  for y in width
    for z in depth

And the 3D dimension is organized as [h, w, d]
*/

namespace utils {
    
  struct Options {
    // Command line arguments (with default values)
    unsigned num_ipus;
    unsigned num_iterations;
    float alpha;
    std::size_t height;
    std::size_t width;
    std::size_t depth;
    std::string vertex;
    bool cpu;
    // Not command line arguments
    std::size_t side;
    std::size_t tiles_per_ipu = 0;
    std::size_t num_tiles_available = 0;
    std::size_t halo_volume = 0;
    std::vector<std::size_t> splits = {0,0,0};
    std::vector<std::size_t> smallest_slice = {std::numeric_limits<size_t>::max(),1,1};
    std::vector<std::size_t> largest_slice = {0,0,0};
  };

  inline
  Options parseOptions(int argc, char** argv) {
    Options options;
    namespace po = boost::program_options;
    po::options_description desc("Flags");
    desc.add_options()
    ("help", "Show command help.")
    (
      "num-ipus",
      po::value<unsigned>(&options.num_ipus)->default_value(1),
      "Number of IPUs (must be a power of 2)"
    )
    (
      "num-iterations",
      po::value<unsigned>(&options.num_iterations)->default_value(100000),
      "PDE: number of iterations to execute on grid."
    )
    (
      "height",
      po::value<std::size_t>(&options.height)->default_value(0),
      "Heigth of a custom 3D grid"
    )
    (
      "width",
      po::value<std::size_t>(&options.width)->default_value(0), 
      "Width of a custom 3D grid"
    )
    (
      "depth",
      po::value<std::size_t>(&options.depth)->default_value(0),
      "Depth of a custom 3D grid"
    )
    (
      "alpha",
      po::value<float>(&options.alpha)->default_value(0.1),
      "PDE: update step size given as a float."
    )
    (
      "vertex",
      po::value<std::string>(&options.vertex)->default_value("HeatEquationOptimized"),
      "Name of vertex (from codelets.cpp) to use for the computation."
    )
    (
      "cpu",
      po::bool_switch(&options.cpu)->default_value(false),
      "Also perform CPU execution to control results from IPU."
    ); // NOTE: remember to remove this semicolon if more options are added in future
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      throw std::runtime_error("Show help");
    }
    po::notify(vm);
    return options;
  }

} // End of namespace Utils

poplar::Device getDevice(utils::Options &options) {
  /* return a Poplar device with the desired number of IPUs */
  std::size_t n = options.num_ipus;
  if (n!=1 && n!=2 && n!=4 && n!=8 && n!=16 && n!=32 && n!=64)
    throw std::runtime_error("Invalid number of IPUs.");

  // Create device manager
  auto manager = poplar::DeviceManager::createDeviceManager();
  auto devices = manager.getDevices(poplar::TargetType::IPU, n);

  // Use the first available device
  for (auto &device : devices)
    if (device.attach()) 
      return std::move(device);

  throw std::runtime_error("No hardware device available.");
}

std::size_t side_length(std::size_t num_ipus, std::size_t base_length) {
  std::size_t log2_num_ipus = log(num_ipus) / log(2);
  std::size_t side = base_length*pow(1.26, log2_num_ipus);
  return side;
}

inline float randomFloat() {
  return static_cast <float> (rand() / static_cast <float> (RAND_MAX));
}

inline static unsigned index(unsigned x, unsigned y, unsigned z, unsigned width, unsigned depth) { 
  return (z) + (y)*(depth) + (x)*(width)*(depth);
}

inline static unsigned block_low(unsigned id, unsigned p, unsigned n) {
  return (id*n)/p; 
}

inline static unsigned block_high(unsigned id, unsigned p, unsigned n) {
  return block_low(id+1, p, n); 
}

inline static unsigned block_size(unsigned id, unsigned p, unsigned n) {
  return block_high(id, p, n) - block_low(id, p, n); 
}

std::size_t volume(std::vector<std::size_t> shape) {
  // return volume of shape vector (3D)
  return shape[0]*shape[1]*shape[2];
}

void workDivision(utils::Options &options) {
  /* Function UPDATES options.splits
   * 1) all tiles will be used, hence options.num_tiles_available must
   *    previously be updated by using the target object
   * 2) the average resulting slice will have shape [height/nh, width/nw, depth/nd'] and
   *    this function chooses nh, nw, nd, so that the surface area is minimized.
   */
  float smallest_surface_area = std::numeric_limits<float>::max();
  std::size_t height = (options.height - 2);
  std::size_t width = (options.width - 2);
  std::size_t depth = (options.depth - 2) / options.num_ipus;
  std::size_t tile_count = options.num_tiles_available / options.num_ipus;
  for (std::size_t i = 1; i <= tile_count; ++i) {
    if (tile_count % i == 0) { // then i is a factor
      // Further, find two other factors, to obtain exactly three factors
      std::size_t other_factor = tile_count/i;
      for (std::size_t j = 1; j <= other_factor; ++j) {
        if (other_factor % j == 0) { // then j is a second factor
          std::size_t k = other_factor/j; // and k is the third factor
          std::vector<std::size_t> splits = {i,j,k}; 
          if (i*j*k != tile_count) {
            throw std::runtime_error("workDivision(), factorization does not work.");
          }
          for (std::size_t l = 0; l < 3; ++l) {
            for (std::size_t m = 0; m < 3; ++m) {
              for (std::size_t n = 0; n < 3; ++n) {
                if (l != m && l != n && m != n) {
                  float slice_height = float(height)/float(splits[l]);
                  float slice_width = float(width)/float(splits[m]);
                  float slice_depth = float(depth)/float(splits[n]);
                  float surface_area = 2.0*(slice_height*slice_width + slice_depth*slice_width + slice_depth*slice_height);
                  if (surface_area <= smallest_surface_area) {
                    smallest_surface_area = surface_area;
                    options.splits = splits;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

std::vector<float> heatEquationCpu(
  const std::vector<float> initial_values, 
  const utils::Options &options) {
  /*
   * Heat Equation Vertex, computed on the CPU, unparallelized.
   * The purpose of this function is to serve as the "true"
   * solution.
   * NOTE: can be very slow for large grids/large number of iterations
   */
  const float beta = 1.0 - 6.0*options.alpha;
  unsigned h = options.height;
  unsigned w = options.width;
  unsigned d = options.depth;
  unsigned iter = options.num_iterations;
  std::vector<float> a(initial_values.size());
  std::vector<float> b(initial_values.size());

  // initial copy to include edges
  for (std::size_t i = 0; i < initial_values.size(); ++i)  {
    a[i] = initial_values[i]; 
    b[i] = initial_values[i]; 
  }

  // Heat Equation iterations
  for (std::size_t t = 0; t < iter; ++t) {
    for (std::size_t x = 1; x < h - 1; ++x) {
      for (std::size_t y = 1; y < w - 1; ++y) { 
        for (std::size_t z = 1; z < d - 1; ++z) {
          a[index(x,y,z,w,d)] = beta*b[index(x,y,z,w,d)] +
            options.alpha*(
              b[index(x+1,y,z,w,d)] +
              b[index(x-1,y,z,w,d)] +
              b[index(x,y+1,z,w,d)] +
              b[index(x,y-1,z,w,d)] +
              b[index(x,y,z+1,w,d)] +
              b[index(x,y,z-1,w,d)]
            );
        }
      }
    }
    for (std::size_t x = 1; x < h - 1; ++x) {
      for (std::size_t y = 1; y < w - 1; ++y) { 
        for (std::size_t z = 1; z < d - 1; ++z) {
          b[index(x,y,z,w,d)] = a[index(x,y,z,w,d)];
        }
      }
    }
  }
  return a;
}

void printMeanSquaredError(
  std::vector<float> a, 
  std::vector<float> b, 
  utils::Options &options) {
  /*
   * Compute the MSE, __only the inner elements__, of two 3D grids
   */
  double squared_error = 0, diff;
  std::size_t h = options.height;
  std::size_t w = options.width;
  std::size_t d = options.depth;
  for (std::size_t x = 1; x < h - 1; ++x) {
    for (std::size_t y = 1; y < w - 1; ++y) { 
      for (std::size_t z = 1; z < d - 1; ++z) {
        diff = double(a[index(x,y,z,w,d)] - b[index(x,y,z,w,d)]);
        squared_error += diff*diff;
      }
    }
  }
  double mean_squared_error = squared_error / (double) ((h-2)*(w-2)*(d-2));

  std::cout << "\nMean Squared Error (IPU vs. CPU) = " << mean_squared_error;
  if (mean_squared_error == double(0.0)) 
    std::cout << " (exactly)";
  std::cout << "\n";
}

void printResults(utils::Options &options, double wall_time) {

  // Calculate metrics
  double inner_volume = (double) options.height * (double) options.width * (double) options.depth;
  double flops_per_element = 8.0;
  double flops = inner_volume * options.num_iterations * flops_per_element / wall_time;
  double internal_communication_ops = 2.0*(double)options.halo_volume*(double)options.num_ipus;
  double external_communication_ops = 4.0*(double)options.height*(double)options.width*(options.num_ipus - 1.0); // 2 load and 2 stores of a slice (partition along depth)
  double bandwidth = (7.0*inner_volume + internal_communication_ops + external_communication_ops)*(double)options.num_iterations*sizeof(float)/wall_time;
  double tflops = flops*1e-12;
  double bandwidth_TB_s = bandwidth*1e-12;

  std::cout << "3D Isotropic Diffusion"
    << "\n----------------------"
    << "\nVertex             = " << options.vertex
    << "\nNo. IPUs           = " << options.num_ipus
    << "\nNo. Tiles          = " << options.num_tiles_available
    << "\nTotal Grid         = " << options.height << "*" << options.width << "*" << options.depth << " = "
                                 << options.height*options.width*options.depth*1e-6 << " million elements"
    << "\nSmallest Sub-grid  = " << options.smallest_slice[0] << "*" << options.smallest_slice[1] << "*" << options.smallest_slice[2] 
    << "\nLargest Sub-grid   = " << options.largest_slice[0] << "*" << options.largest_slice[1] << "*" << options.largest_slice[2] 
    << "\nalpha              = " << options.alpha
    << "\nNo. Iterations     = " << options.num_iterations
    << "\n"
    << "\nLaTeX Tabular Row"
    << "\n-----------------"
    << "\nNo. IPUs & Grid & No. Iterations & Time [s] & Throughput [TFLOPS] & Minimum Bandwidth [TB/s] \\\\\n" 
    << options.num_ipus << " & "
    << "$" << options.height << "\\times " << options.width << "\\times " << options.depth << "$ & " 
    << options.num_iterations << " & " << std::fixed
    << std::setprecision(2) << wall_time << " & " 
    << std::setprecision(2) << tflops << " & " 
    << std::setprecision(2) << bandwidth_TB_s << " \\\\"
    << "\n";
}

void printMultiIpuGridInfo(std::size_t base_length) {
  float base_volume = 0;

  std::cout 
    << "\\begin{tabular}{ccc}"
    << "\n\\toprule"
    << "\nNumber of IPUs & Grid Shape & Relative Volume \\\\\\midrule"
    << std::fixed << std::setprecision(2);

  for (std::size_t i = 1; i <= 64; i*=2) {
    std::size_t side = side_length(i, base_length);
    float volume = float(side)*float(side)*float(side);
    if (i == 1) base_volume = volume;
    std::cout << "\n" 
      << i << " & $"
      << side << "\\times " << side << "\\times " << side << "$ & "
      << volume/base_volume << "x \\\\";
  }

  std::cout 
    << "\n\\bottomrule"
    << "\n\\end{tabular}\n";
}
