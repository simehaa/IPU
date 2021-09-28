#pragma once
#include <boost/program_options.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <fstream>
#include <math.h>

/*
README
To avoid confusion in indexing:
  x goes along heigth
  y goes along width
*/

namespace utils {

  struct Options {
    // Command line arguments (with default values)
    unsigned num_ipus;
    unsigned height;
    unsigned width;
    unsigned num_iterations;
    float my1;
    float my2;
    float k;
    float epsilon;
    float b;
    float a;
    float h;
    float dt;
    float delta;
    bool cpu;
    bool save_cpu;
    // Other arguments which are a consequence of the code or environment
    std::string architecture; // Assigned when creating device
    unsigned num_tiles_available = 0;
    std::size_t halo_volume = 0;
    unsigned nh = 0;
    unsigned nw = 0;
    float upper_bound_dt;
    std::vector<std::size_t> smallest_slice = {std::numeric_limits<size_t>::max(),1};
    std::vector<std::size_t> largest_slice = {0,0};
  };

  inline
  Options parseOptions(int argc, char** argv) {
    Options options;
    namespace po = boost::program_options;
    po::options_description desc("Flags");
    // Construction of exactly cubic sub-grids
    desc.add_options()
    ("help", "Show command help.")
    (
      "num-ipus",
      po::value<unsigned>(&options.num_ipus)->default_value(1),
      "Number of IPUs to use."
    )
    (
      "height", 
      po::value<unsigned>(&options.height)->default_value(7000),
      "Heigth of a custom 2D grid."
    )
    (
      "width",
      po::value<unsigned>(&options.width)->default_value(7000),
      "Width of a custom 2D grid."
    )
    (
      "num-iterations",
      po::value<unsigned>(&options.num_iterations)->default_value(1000),
      "PDE: number of iterations to execute on grid."
    )
    (
      "my1",
      po::value<float>(&options.my1)->default_value(0.07),
      "A constant in the forward Euler Aliev-Panfilov equations."
    )
    (
      "my2",
      po::value<float>(&options.my2)->default_value(0.3),
      "A constant in the forward Euler Aliev-Panfilov equations."
    )
    (
      "k",
      po::value<float>(&options.k)->default_value(8.0),
      "A constant in the forward Euler Aliev-Panfilov equations."
    )
    (
      "epsilon",
      po::value<float>(&options.epsilon)->default_value(0.01),
      "A constant in the forward Euler Aliev-Panfilov equations."
    )
    (
      "b",
      po::value<float>(&options.b)->default_value(0.1),
      "A constant in the forward Euler Aliev-Panfilov equations."
    )
    (
      "a",
      po::value<float>(&options.a)->default_value(0.1),
      "A constant in the forward Euler Aliev-Panfilov equations."
    )
    (
      "dt",
      po::value<float>(&options.dt)->default_value(0.0001),
      "A constant in the forward Euler Aliev-Panfilov equations."
    )
    (
      "h",
      po::value<float>(&options.h)->default_value(0.000143),
      "A constant in the forward Euler Aliev-Panfilov equations."
    )
    (
      "delta",
      po::value<float>(&options.delta)->default_value(5.0e-5),
      "A constant in the forward Euler Aliev-Panfilov equations."
    )
    (
      "cpu",
      po::bool_switch(&options.cpu)->default_value(false),
      "Also perform CPU execution to control results from IPU."
    )
    (
      "save-cpu",
      po::bool_switch(&options.save_cpu)->default_value(false),
      "Save CPU results to csv files in data folder."
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

inline static unsigned index(unsigned x, unsigned y, unsigned width) { 
  return y + (x*width);
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

std::size_t area(std::vector<std::size_t> shape) {
  // return area of shape vector (2D)
  return shape[0]*shape[1];
}

void workDivision(utils::Options &options) {
  /* Function UPDATES options.nh and options.nw
   * nh and nw will be chosen so that
   * 1) all tiles will be used, hence options.num_tiles_available must
   *    previously be updated by using the target object
   * 2) find work division which results in most square patches (sub-grids)
   */
  unsigned tile_count = options.num_tiles_available;
  unsigned H = options.height - 2; // Actual height (boundaries wont be computed)
  unsigned W = options.width - 2; // Actual width
  float best_patch_ratio = std::numeric_limits<float>::infinity();
  bool height_dominant = (H >= W);

  // Try all unique combinations where i*j = tile_count
  for (unsigned i = 1; i*i <= tile_count; ++i) {
    if ((tile_count % i) == 0) {
      unsigned j = tile_count / i; // j >= i
      unsigned nh = height_dominant ? j : i;
      unsigned nw = height_dominant ? i : j;

      unsigned h = float(H) / nh;
      unsigned w = float(W) / nw;
      // largest side length / smallest side length
      // this will result in a ratio >= 1, where the ratio
      // which is closest to 1, has the most square patches (1=perfect square)
      float patch_ratio = (h > w) ? float(h)/w : float(w)/h;
      if (patch_ratio < best_patch_ratio) {
        best_patch_ratio = patch_ratio;
        options.nh = nh;
        options.nw = nw;
      }
    }
  }

  if (options.nw == 0 || options.nh == 0) {
    std::cout << "Work division went wrong. Using 1 tile.\n";
    options.nh = 1;
    options.nw = 1;
  }
}

void print2dVector(std::vector<float> grid, utils::Options &options, std::string name) {
  int h = options.height;
  int w = options.width;
  std::cout << name << ":\n";
  for (int i = 0; i < h; ++i) {
    if (i == 0) {
      std::cout << "[";
    } else {
      std::cout << " ";
    }
    for (int j = 0; j < w; ++j) {
      if (j == 0) std::cout << "["; 
      std::cout << grid[j + i*w];
      if (j < w - 1) {
        std::cout << ", ";
      } else {
        std::cout << "]";
      }
    }
    if (i < h - 1) {
      std::cout << "\n";
    } else {
      std::cout << "]\n\n";
    }
  }
}

void save2dVector(std::vector<float> grid, utils::Options &options, std::string filename) {
  int h = options.height;
  int w = options.width;
  std::ofstream outfile;
  outfile.open(filename);
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      outfile << grid[j + i*w];
      if (j < w - 1)
        outfile << ",";
    }
    if (i < h - 1) outfile << "\n";
  }
  outfile.close();
}

void solveAlievPanfilovCpu(
  const std::vector<float> initial_e, const std::vector<float> initial_r, 
  std::vector<float> &cpu_e, std::vector<float> &cpu_r, utils::Options &options) {

  int N = initial_e.size();
  int w = options.width;
  int h = options.height;
  float rhs;
  std::vector<float> temp_e(N);
  for (int i = 0 ; i < N; ++i) {
    cpu_e[i] = initial_e[i];
    cpu_r[i] = initial_r[i];
    temp_e[i] = initial_e[i];
  }
  const float d_h2 = options.delta/(options.h*options.h);
  const float minus_epsilon = -options.epsilon;
  const float b_plus_one = options.b + 1;
  float west, north, east, south;
  int c = 0;

  for (int t = 0; t < options.num_iterations; ++t) {
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        
        // Boundary condition (zero gradient)
        west = (j == 0) ? cpu_e[(j+1)+i*w] : cpu_e[(j-1)+i*w];
        east = (j == w - 1) ? cpu_e[(j-1)+i*w] : cpu_e[(j+1)+i*w];
        north = (i == 0) ? cpu_e[j+(i+1)*w] : cpu_e[j+(i-1)*w];
        south = (i == h - 1) ? cpu_e[j+(i-1)*w] : cpu_e[j+(i+1)*w];
        
        // Computation of new e
        rhs = d_h2*(-4*cpu_e[j+i*w] + west + east + south + north);
        rhs -= options.k*cpu_e[j+i*w]*(cpu_e[j+i*w] - options.a)*(cpu_e[j+i*w] - 1);
        rhs -= cpu_e[j+i*w]*cpu_r[j+i*w];
        temp_e[j+i*w] = cpu_e[j+i*w] + rhs*options.dt;

        // Computation of new r
        rhs = minus_epsilon - options.my1*cpu_r[j+i*w]/(options.my2 + cpu_e[j+i*w]);
        rhs *= cpu_r[j+i*w] + options.k*cpu_e[j+i*w]*(cpu_e[j+i*w] - b_plus_one);
        cpu_r[j+i*w] += rhs*options.dt;
      }
    }
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        cpu_e[j+i*w] = temp_e[j+i*w];
      }
    }
    if (t % 500 == 0 && options.save_cpu) {
      save2dVector(cpu_e, options, "./data/e"+std::to_string(c)+".csv");
      save2dVector(cpu_r, options, "./data/r"+std::to_string(c)+".csv");
      c++; // nice
    }
  }
}

void printGeneralInfo(utils::Options &options) {
  std::cout
    << "\nAliev-Panfilov Forward Euler Method"
    << "\n-----------------------------------"
    << "\n\nProblem"
    << "\n-------"
    << "\n2D Grids = " << options.height << "x" << options.width << " elements"
    << "\nWork Division = " << options.nh << "x" << options.nw << " partitions"
    << "\nNo. Time Steps = " << options.num_iterations
    << "\n\nConstants"
    << "\n---------"
    << "\ndelta = " << options.delta
    << "\nmy1 = " << options.my1
    << "\nmy2 = " << options.my2
    << "\na = " << options.a
    << "\nb = " << options.b
    << "\nk = " << options.k
    << "\ndx = " << options.h
    << "\ndy = " << options.h
    << "\ndt = " << options.dt << " (upper bound = " << options.upper_bound_dt << ")"
    << "\nepsilon = " << options.epsilon
    << "\n\n";
}

void testUpperBoundDt(utils::Options &options) {
  float ka = options.k*options.a;
  float k_1_a = options.k*(1 - options.a);
  float max = (ka > k_1_a) ? ka : k_1_a;
  float lambda = options.delta/(options.h*options.h);
  float r_plus = options.k*(options.b+1)*(options.b+1)/4.0;
  options.upper_bound_dt = 1.0/(4*lambda + max + r_plus);
  if (options.dt > options.upper_bound_dt) 
    throw std::runtime_error(
      "Forward Euler method is not stable, because dt ("+std::to_string(options.dt)+
      ") > upper bound ("+std::to_string(options.upper_bound_dt)+")."
    );
}

void reportCpuVsIpu(std::vector<float> cpu_e, std::vector<float> cpu_r, 
  std::vector<float> ipu_e, std::vector<float> ipu_r, utils::Options &options) {
  
  std::size_t w = options.width;
  std::size_t h = options.height;
  float MSE_e = 0;
  float MSE_r = 0;
  for (int i = 0; i < options.height; ++i) {
    for (int j = 0; j < options.width; ++j) {
      float diff_e = cpu_e[j + i*w] - ipu_e[j + i*w];
      float diff_r = cpu_r[j + i*w] - ipu_r[j + i*w];
      MSE_e += diff_e*diff_e;
      MSE_r += diff_r*diff_r;
    }
  }
  MSE_e /= w*h;
  MSE_r /= w*h;
  std::cout
    << "\nAliev-Panfilov IPU vs. CPU error"
    << "\n--------------------------------"
    << "\nMSE of e = " << MSE_e;
  if (MSE_e == 0) std::cout << " (exactly)";
  std::cout << "\nMSE of r = " << MSE_r;
  if (MSE_r == 0) std::cout << " (exactly)";
  std::cout << "\n\n";
}

void printPerformance(double wall_time, utils::Options &options) {
  double flops_per_element = 28.0;
  double elements_per_time = (double) options.height * (double) options.width * (double) options.num_iterations / (double) wall_time;
  double flops = flops_per_element * elements_per_time;
  double comp_mem_bw = 8*elements_per_time*sizeof(float); // 6 loads (5-point stencil for e, and r) + 2 stores (e and r)
  double comm_mem_bw = 2.0*(double)options.halo_volume*sizeof(float)*(double) options.num_iterations / (double) wall_time; 
  double minimal_bw = comp_mem_bw + comm_mem_bw;
  std::cout
    << "\nPerformance"
    << "\n-----------" << std::fixed
    << "\nTime       = " << std::setprecision(2) << wall_time << " s"
    << "\nThroughput = " << std::setprecision(2) << flops*1e-12 << " TFLOPS"
    << "\nMinimal BW = " << std::setprecision(2) << minimal_bw*1e-12 << " TB/s"
    << "\n\n";
}