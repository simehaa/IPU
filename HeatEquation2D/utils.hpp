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
    unsigned height;
    unsigned width;
    unsigned num_iterations;
    float alpha;
    bool cpu;
    bool random_tiles;
    std::string in_file;
    std::string out_file;
    // Other arguments which are a consequence of the code or environment
    std::string architecture; // Assigned when creating device
    unsigned num_tiles = 0;
    unsigned nh = 0;
    unsigned nw = 0;
    unsigned num_tiles_available = 0;
    unsigned largest_tile_height = 0;
    unsigned largest_tile_width = 0;
    unsigned largest_tile_area = 0;
    unsigned smallest_tile_height = UINT_MAX; // = 4,294,967,295 (should be safe)
    unsigned smallest_tile_width = UINT_MAX;
    unsigned smallest_tile_area = UINT_MAX;
    std::vector<int> random_perm;
    float memory_available;
    bool first_compute_set = true;
    denoise::Image image; // copy of image (if input jpg image is used)
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
      po::value<unsigned>(&options.height)->default_value(46*204+2),
      "Heigth of a custom 2D grid. Will be overwritten if --in-file is used."
    )
    (
      "width",
      po::value<unsigned>(&options.width)->default_value(32*204+2),
      "Width of a custom 2D grid. Will be overwritten if --in-file is used."
    )
    (
      "num-iterations",
      po::value<unsigned>(&options.num_iterations)->default_value(100),
      "PDE: number of iterations to execute on grid."
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
      "random-tiles",
      po::bool_switch(&options.random_tiles)->default_value(false),
      "Assign tile mappings randomly among the tiles."
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
    std::cerr << "Failed to write: " << options.out_file << std::endl;
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

double meanSquaredErrorInnerElements(
  std::vector<float> a, 
  std::vector<float> b, 
  utils::Options &options) {
  /*
   * Compute the MSE on only the inner elements of two 2D grids
   */
  double squared_error = 0;
  double diff;
  unsigned h = options.height;
  unsigned w = options.width;
  for (std::size_t x = 1; x < h - 1; ++x) {
    for (std::size_t y = 1; y < w - 1; ++y) { 
      diff = double(a[index(x,y,w)] - b[index(x,y,w)]);
      squared_error += diff*diff;
    }
  }
  double inner_area = double((h-2)*(w-2));
  double mean_squared_error = squared_error / inner_area;
  return mean_squared_error;
}

void printGeneralInfo(utils::Options &options) {

  // Calculate metrics
  double total_area = options.height * options.width;
  double memory_used = total_area * sizeof(float) * 2;
  double memory_fraction = total_area * sizeof(float) * 2 / float(options.memory_available);
  double tile_balance = 100 * float(options.smallest_tile_area) / float(options.largest_tile_area);

  std::cout 
    << "\n2D Isotropic Diffusion"
    << "\n----------------------"
    << "\n"
    << "\nParameters"
    << "\n----------"
    << "\nIPU Architecture    = " << options.architecture
    << "\nTotal Grid          = " << options.height << "*" << options.width << " = "
                                  << total_area*1e-6 << " million elements"
    << "\nLargest Sub-grid    = " << options.largest_tile_height << "*" << options.largest_tile_width << " = " 
                                  << options.largest_tile_area << " elements"
    << "\nSmallest Sub-grid   = " << options.smallest_tile_height << "*" << options.smallest_tile_width << " = " 
                                  << options.smallest_tile_area << " elements"
    << "\nTile Balance        = " << tile_balance << " %"
    << "\nMemory (2x tensors) = " << memory_used * 1e-6 << " MB (" << memory_fraction * 100 << " \% of " << options.memory_available * 1e-6 << " MB)"
    << "\nAlpha               = " << options.alpha
    << "\nNum. Iterations     = " << options.num_iterations
    << "\nNum. IPUs Used      = " << options.num_ipus
    << "\nNum. Tiles Used     = " << options.num_tiles << " (" << options.num_tiles_available << " available)"
    << "\nWork Division       = " << options.nh << "*" << options.nw
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
      << " & " << std::setprecision(3) << 5*bandwidth_TB_S[i] << " TB/s"
      << " & " << std::setprecision(3) << 6*bandwidth_TB_S[i] << " TB/s"
      << " \\\\\n";
  }
  
  // End of tabular
  std::cout << "\\bottomrule\n\\end{tabular}\n"; 
}