#pragma once
#include <boost/program_options.hpp>
#include <string>
#include <iostream>
#include <iomanip>
#include <math.h>

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
    unsigned height;
    unsigned width;
    unsigned depth;
    unsigned num_iterations;
    float alpha;
    bool cpu;
    bool random_tiles;
    std::string vertex;
    // Other arguments which are a consequence of the code or environment
    std::string architecture; // Assigned when creating device
    std::size_t num_tiles = 0;
    std::size_t nh = 0;
    std::size_t nw = 0;
    std::size_t nd = 0;
    std::size_t num_tiles_available = 0;
    std::size_t largest_tile_height = 0;
    std::size_t largest_tile_width = 0;
    std::size_t largest_tile_depth = 0;
    std::size_t largest_tile_volume = 0;
    std::size_t smallest_tile_height = UINT_MAX; // = 4,294,967,295 (should be safe)
    std::size_t smallest_tile_width = UINT_MAX;
    std::size_t smallest_tile_depth = UINT_MAX;
    std::size_t smallest_tile_volume = UINT_MAX;
    std::vector<int> random_perm;
    float memory_available;
    bool first_compute_set = true;
  };

  std::vector<std::string> getVertices() {
    // Names of vertices to include in computation
    std::vector<std::string> vertices = {
      "HeatEquationSimple",
      "HeatEquationOptimized"
    };
    return vertices;
  }

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
      "Number of IPUs to use."
    )
    (
      "height",
      po::value<unsigned>(&options.height)->default_value(8*24+2),
      "Heigth of a custom 3D grid"
    )
    (
      "width",
      po::value<unsigned>(&options.width)->default_value(8*24+2), 
      "Width of a custom 3D grid"
    )
    (
      "depth",
      po::value<unsigned>(&options.depth)->default_value(23*24+2),
      "Depth of a custom 3D grid"
    )
    (
      "num-iterations",
      po::value<unsigned>(&options.num_iterations)->default_value(100),
      "PDE: number of iterations to execute on grid."
    )
    (
      "alpha",
      po::value<float>(&options.alpha)->default_value(0.1),
      "PDE: update step size given as a float."
    )
    (
      "cpu",
      po::bool_switch(&options.cpu)->default_value(false),
      "Also perform CPU execution to control results from IPU."
    )
    (
      "random-tiles",
      po::bool_switch(&options.random_tiles)->default_value(false),
      "Assign tile mappings randomly among the tiles."
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

poplar::Device getDevice(unsigned numIpus) {
  /* return a Poplar device with the desired number of IPUs */
  auto manager = poplar::DeviceManager::createDeviceManager();
  auto devices = manager.getDevices(poplar::TargetType::IPU, numIpus);
  // Use the first available device
  for (auto &device : devices)
    if (device.attach()) 
      return std::move(device);

  throw std::runtime_error("No hardware device available.");
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

std::vector<int> primeFactorization(int num) {
  
} 

void workDivision(utils::Options &options) {
  /* Function UPDATES options.nh, options.nw, and options.nd
   * The average resulting slice will have shape [height/nh, width/nw, depth/nd']
   * This function chooses nh, nw, nd, so that the surface area is minimized.
   */
  float smallest_surface_area = std::numeric_limits<float>::max();
  int tile_count = options.num_tiles_available;
  for (int i = 1; i <= tile_count; ++i) {
    if (tile_count % i == 0) { // then i is a factor
      // Further, find two other factors, to obtain exactly three factors
      int other_factor = tile_count/i;
      for (int j = 1; j <= other_factor; ++j) {
        if (other_factor % j == 0) { // then j is a second factor
          int k = other_factor/j; // and k is the third factor
          std::vector<int> splits = {i,j,k}; 
          if (i*j*k != tile_count) {
            throw std::runtime_error("workDivision(), factorization does not work.");
          }
          for (int l = 0; l < 3; ++l) {
            for (int m = 0; m < 3; ++m) {
              for (int n = 0; n < 3; ++n) {
                if (l != m && l != n && m != n) {
                  float slice_height = float(options.height)/float(splits[l]);
                  float slice_width = float(options.width)/float(splits[m]);
                  float slice_depth = float(options.depth)/float(splits[n]);
                  float surface_area = 2.0*(slice_height*slice_width + slice_depth*slice_width + slice_depth*slice_height);
                  if (surface_area <= smallest_surface_area) {
                    smallest_surface_area = surface_area;
                    options.nh = splits[l];
                    options.nw = splits[m];
                    options.nd = splits[n];
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

void print_3d_variable(std::vector<float> values, const utils::Options &options, bool inner_only) {
  /*
   * Utility function to print a 3D variable which is stored as a 1D vector with the index 
   * paradigm (z) + (y)*(depth) + (x)*(width)*(depth).
   */
  // Loop over slices
  int offset = 0;
  if (inner_only) 
    offset = 1;
  
  for (std::size_t z = 0 + offset; z < options.depth - offset; ++z) {
    std::cout << "Slice " << z << ":\n[";
    // Loop over rows (height dim)
    for (std::size_t x = 0 + offset; x < options.height - offset; ++x) {
      if (x != 0 + offset) // space for alignment
        std::cout << " ";
      std::cout << "["; // start of row
      // Loop over columns
      for (std::size_t y = 0 + offset; y < options.width - offset; ++y) {
        std::cout << values[index(x,y,z,options.width,options.depth)];
        if (y < options.width - 1 - offset) 
          std::cout << ", "; // Comma separate inner values
      }
      std::cout << "]"; // end of row
      if (x < options.height - 1 - offset) 
        std::cout << "\n"; // new line to next row
    }
    std::cout << "]\n"; // new line to next slice (last row)
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

double meanSquaredErrorInnerElements(
  std::vector<float> a, 
  std::vector<float> b, 
  utils::Options &options) {
  /*
   * Compute the MSE on only the inner elements of two 3D grids
   */
  double squared_error = 0, diff;
  unsigned h = options.height;
  unsigned w = options.width;
  unsigned d = options.depth;
  for (std::size_t x = 1; x < h - 1; ++x) {
    for (std::size_t y = 1; y < w - 1; ++y) { 
      for (std::size_t z = 1; z < d - 1; ++z) {
        diff = double(a[index(x,y,z,w,d)] - b[index(x,y,z,w,d)]);
        squared_error += diff*diff;
      }
    }
  }
  double mean_squared_error = squared_error / double((h-2)*(w-2)*(d-2));
  return mean_squared_error;
}

void printGeneralInfo(utils::Options &options) {

  // Calculate metrics
  double total_volume = options.height * options.width * options.depth;
  double memory_used = total_volume * sizeof(float) * 2;
  double memory_fraction = total_volume * sizeof(float) * 2 / float(options.memory_available);
  double tile_balance = 100 * float(options.smallest_tile_volume) / float(options.largest_tile_volume);

  std::cout 
    << "\n3D Isotropic Diffusion"
    << "\n----------------------"
    << "\n"
    << "\nParameters"
    << "\n----------"
    << "\nIPU Architecture    = " << options.architecture
    << "\nTotal Grid          = " << options.height << "*" << options.width << "*" << options.depth << " = "
                                  << total_volume*1e-6 << " million elements"
    << "\nLargest Sub-grid    = " << options.largest_tile_height << "*" << options.largest_tile_width << "*" 
                                  << options.largest_tile_depth << " = " << options.largest_tile_volume << " elements"
    << "\nSmallest Sub-grid   = " << options.smallest_tile_height << "*" << options.smallest_tile_width << "*" 
                                  << options.smallest_tile_depth << " = " << options.smallest_tile_volume << " elements"
    << "\nTile Balance        = " << tile_balance << " %"
    << "\nMemory (2x tensors) = " << memory_used * 1e-6 << " MB (" << memory_fraction * 100 << " \% of " << options.memory_available * 1e-6 << " MB)"
    << "\nAlpha               = " << options.alpha
    << "\nNum. Iterations     = " << options.num_iterations
    << "\nNum. IPUs Used      = " << options.num_ipus
    << "\nNum. Tiles Used     = " << options.num_tiles << " (" << options.num_tiles_available << " available)"
    << "\nWork Division       = " << options.nh << "*" << options.nw << "*" << options.nd
    << "\nRandom Tiles        = " << options.random_tiles;
    
  std::cout << "\n";
}

void printMeanSquaredError(
  const std::vector<std::string> vertices, 
  const std::vector<double> MSE) {
  /* Print MSE of IPU results vs. CPU results */
  if (vertices.size() > 0 && MSE.size() > 0 && vertices.size() == MSE.size()) {
    std::cout 
      << "\nMean Squared Error (IPU vs. CPU)"
      << "\n--------------------------------";
    // Add rows (one value per vertex)
    for (std::size_t i = 0; i < MSE.size(); ++i) {
      std::cout << "\n" << vertices[i] << " = " << MSE[i];
      if (MSE[i] == double(0.0)) 
        std::cout << " (exactly)";
    }
    std::cout << "\n";
  }
}

void printLatexTabular(
    std::vector<std::string> vertices,
    std::vector<double> TFLOPS,
    std::vector<double> bandwidth_TB_S) {

  // LaTeX tabular print
  std::cout
    << "\nLaTeX Tabular" 
    << "\n-------------"
    << "\n\\begin{tabular}{lllll}"
    << "\n\\toprule"
    << "\nVertex & TFLOPS & Store BW & Load BW & Total BW \\\\\\midrule\n" << std::fixed;

  // Add rows with data
  for (std::size_t i = 0; i < vertices.size(); ++i) {
    std::cout << "\\mintinline{c++}{" << vertices[i] << "}"
      << " & " << std::setprecision(3) << TFLOPS[i] 
      << " & " << std::setprecision(3) << bandwidth_TB_S[i] << " TB/s"
      << " & " << std::setprecision(3) << 7*bandwidth_TB_S[i] << " TB/s"
      << " & " << std::setprecision(3) << 8*bandwidth_TB_S[i] << " TB/s"
      << " \\\\\n";
  }
  
  // End of tabular
  std::cout << "\\bottomrule\n\\end{tabular}\n"; 
}