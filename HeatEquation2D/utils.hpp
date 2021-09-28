#pragma once
#include <boost/program_options.hpp>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include "image.pb.h"

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
    unsigned num_iterations;
    float alpha;
    std::size_t height;
    std::size_t width;
    std::string vertex;
    std::string in_file;
    std::string out_file;
    bool cpu;
    // Other arguments which are a consequence of the code or environment
    std::size_t side;
    std::size_t tiles_per_ipu = 0;
    std::size_t num_tiles_available = 0;
    std::size_t halo_volume = 0;
    std::vector<std::size_t> splits = {0,0};
    std::vector<std::size_t> smallest_slice = {std::numeric_limits<size_t>::max(),1};
    std::vector<std::size_t> largest_slice = {0,0};
    denoise::Image image; // copy of image (if input jpg image is used)
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
      "num-iterations",
      po::value<unsigned>(&options.num_iterations)->default_value(1000),
      "PDE: number of iterations to execute on grid."
    )
    (
      "height", 
      po::value<std::size_t>(&options.height)->default_value(8000),
      "Heigth of a custom 2D grid. Will be overwritten if --in-file is used."
    )
    (
      "width",
      po::value<std::size_t>(&options.width)->default_value(8000),
      "Width of a custom 2D grid. Will be overwritten if --in-file is used."
    )
    (
      "alpha",
      po::value<float>(&options.alpha)->default_value(0.1),
      "PDE: float. kappa * delta t / h^2"
    )
    (
      "cpu",
      po::bool_switch(&options.cpu)->default_value(false),
      "Also perform CPU execution to control results from IPU."
    )
    (
      "vertex",
      po::value<std::string>(&options.vertex)->default_value("HeatEquationOptimized"),
      "Name of vertex (from codelets.cpp) to use for the computation."
    )
    (
      "in-file",
      po::value<std::string>(&options.in_file)->default_value(""),
      "Filename of image (.bin): if empty string, then a random grid will be generated instead."
    )
    (
      "out-file",
      po::value<std::string>(&options.out_file)->default_value("denoised.jpg.bin"),
      "Output jpg file."
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
  /* Function UPDATES options.splits
   * 1) all tiles will be used, hence options.num_tiles_available must
   *    previously be updated by using the target object
   * 2) choose the splits that minimizes the halo regions
   */
  std::size_t tile_count = options.tiles_per_ipu;
  std::size_t height = (options.height - 2); // Actual height (boundaries wont be computed)
  std::size_t width = (options.width - 2) / options.num_ipus; // Actual width
  float smallest_halo_region = std::numeric_limits<float>::infinity();

  // Try all unique combinations where i*j = tile_count
  for (std::size_t i = 1; i <= tile_count; ++i) {
    if ((tile_count % i) == 0) { // then i is a factor
      std::size_t j = tile_count / i; // and j must be the other factor
      std::size_t slice_height = height / i;
      std::size_t slice_width = width / j;
      std::size_t halo_region = 2.0*(slice_height + slice_width) - 4;
      if (halo_region <= smallest_halo_region) {
        smallest_halo_region = halo_region;
        options.splits[0] = i;
        options.splits[1] = j;
      }
    }
  }
}

std::vector<float> imageToVector(const denoise::Image& image) {
  /* Convert data from image into a vector<float> */ 
  std::vector<float> values(image.values().size());
  auto bytes = reinterpret_cast<const unsigned char *>(image.values().c_str());
  for (int i = 0; i < values.size(); ++i)
    values[i] = bytes[i]/255.0f;

  return values;
}

void vectorToImage(utils::Options &options, std::vector<float> &values) {
  /* Convert data from a vector<float> into image*/ 
  auto output = reinterpret_cast<unsigned char *>(&(*options.image.mutable_values())[0]);
  for (int i = 0; i < values.size(); ++i) {
    output[i] = 255.0f*values[i];
  }
}

std::vector<float> initializeValues(utils::Options &options) {
  if (options.in_file.length() != 0) {
    // Load jpg image, convert to vector, and update height and width
    std::fstream image_pb(options.in_file, std::ios::in | std::ios::binary);
    if (!options.image.ParseFromIstream(&image_pb))
      std::cerr << "Failed to read: " << options.in_file << std::endl;
    options.height = options.image.shape(1);
    options.width = options.image.shape(2);
    return imageToVector(options.image);
  } else {
    // Initialize random values
    std::vector<float> initial_values(options.height*options.width);
    for (std::size_t i = 0; i < options.height*options.width; ++i)
      initial_values[i] = randomFloat();
    return initial_values;
  }
}

void saveImage(utils::Options &options, std::vector<float> results) {
  vectorToImage(options, results);
  std::fstream denoised_pb(options.out_file, std::ios::out | std::ios::trunc | std::ios::binary);
  if (!options.image.SerializeToOstream(&denoised_pb)) {
    std::cerr << "Failed to write: " << options.out_file << "\n";
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
   * Approach:
   * for iter = 0, maxIter:
   *    - Heat Eqaution on inner elements of initalValues -> Results
   *    - Copy inner elements of results -> initial_values
   */
  const float gamma = 1.0 - 4.0*options.alpha;
  unsigned h = options.height;
  unsigned w = options.width;
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
        a[index(x,y,w)] = gamma*b[index(x,y,w)] +
          options.alpha*(
            b[index(x+1,y,w)] +
            b[index(x-1,y,w)] +
            b[index(x,y+1,w)] +
            b[index(x,y-1,w)]
          );
      }
    }
    for (std::size_t x = 1; x < h - 1; ++x) {
      for (std::size_t y = 1; y < w - 1; ++y) { 
        b[index(x,y,w)] = a[index(x,y,w)];
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
   * Compute the MSE on only the inner elements of two 2D grids
   */
  double squared_error = 0, diff;
  std::size_t h = options.height;
  std::size_t w = options.width;
  for (std::size_t x = 1; x < h - 1; ++x) {
    for (std::size_t y = 1; y < w - 1; ++y) { 
      diff = double(a[index(x,y,w)] - b[index(x,y,w)]);
      squared_error += diff*diff;
    }
  }
  double mean_squared_error = squared_error / (double) ((h-2)*(w-2));
  
  std::cout << "\nMean Squared Error (IPU vs. CPU) = " << mean_squared_error;
  if (mean_squared_error == double(0.0)) 
    std::cout << " (exactly)";
  std::cout << "\n";
}

void printResults(utils::Options &options, double wall_time) {

  // Calculate metrics
  double inner_area = (double)(options.height - 2)*(double)(options.width - 2);
  double flops_per_element = 6.0;
  double flops = inner_area*options.num_iterations*flops_per_element / wall_time;
  double internal_communication_ops = 2.0*(double)options.halo_volume*(double)options.num_ipus;
  double external_communication_ops = 4.0*(double)options.height*(options.num_ipus - 1.0); // 2 load and 2 stores of a height (partition along width)
  double bandwidth = (5.0*inner_area + internal_communication_ops + external_communication_ops)*(double)options.num_iterations*sizeof(float)/wall_time;
  double tflops = flops*1e-12;
  double bandwidth_TB_s = bandwidth*1e-12;

  std::cout << "2D Isotropic Diffusion"
    << "\n----------------------"
    << "\nVertex             = " << options.vertex
    << "\nNo. IPUs           = " << options.num_ipus
    << "\nNo. Tiles          = " << options.num_tiles_available
    << "\nTotal Grid         = " << options.height << "*" << options.width << " = "
                                 << options.height*options.width*1e-6 << " million elements"
    << "\nSmallest tile grid = " << options.smallest_slice[0] << "*" << options.smallest_slice[1]
    << "\nLargest tile grid  = " << options.largest_slice[0] << "*" << options.largest_slice[1]
    << "\nalpha              = " << options.alpha
    << "\nNo. Iterations     = " << options.num_iterations
    << "\n"
    << "\nLaTeX Tabular Row"
    << "\n-----------------"
    << "\nNo. IPUs & Grid & No. Iterations & Time [s] & Throughput [TFLOPS] & Minimum Bandwidth [TB/s] \\\\\n" 
    << options.num_ipus << " & "
    << "$" << options.height << "\\times " << options.width << "$ & "  
    << options.num_iterations << " & " << std::fixed
    << std::setprecision(2) << wall_time << " & " 
    << std::setprecision(2) << tflops << " & " 
    << std::setprecision(2) << bandwidth_TB_s << " \\\\"
    << "\n";
}