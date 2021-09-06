#include <poplar/Vertex.hpp>
#include <ipudef.h> // for float2

using namespace poplar;

class HeatEquationSimple : public Vertex {
public: 
  HeatEquationSimple();

  Vector<Input<Vector<float, VectorLayout::SPAN, 4, false>>> in;
  Vector<Output<Vector<float, VectorLayout::SPAN, 4, false>>> out;
  const int worker_height;
  const int worker_width;
  const float alpha;
  
  bool compute () {
    const float beta{1.0f - 4.0f*alpha};
    for (int i = 1; i < worker_height + 1; ++i) {
      for (int j = 1; j < worker_width + 1; ++j) {
        out[i-1][j-1] = beta*in[i][j] + alpha*(in[i-1][j] + in[i+1][j] + in[i][j-1] + in[i][j+1]);
      }
    }
    return true;
  }
};

class HeatEquationOptimized : public Vertex {
public: 
  HeatEquationOptimized();
  
  Vector<Input<Vector<float, VectorLayout::SPAN, 8, false>>> in;
  Vector<Output<Vector<float, VectorLayout::SPAN, 4, false>>> out;
  const int worker_height;
  const int worker_width;
  const float alpha;
  
  bool compute () {
    const int half_width = worker_width/2 + (worker_width % 2);
    const float beta{1.0f - 4.0f*alpha};
    typedef float float2 __attribute__((ext_vector_type(2)));
    float2 temp; // Temporary variable
    
    // Loop over rows, only update left/right column
    for (int i = 1; i < worker_height + 1; ++i) {
      // Left column
      int j = 1; 
      out[i-1][j-1] = beta*in[i][j] + alpha*(in[i+1][j] + in[i-1][j] + in[i][j+1] + in[i][j-1]);
      
      // Right column (only if worker_width is even)
      if (worker_width % 2 == 0) {
        j = worker_width; // right column
        out[i-1][j-1] = beta*in[i][j] + alpha*(in[i+1][j] + in[i-1][j] + in[i][j+1] + in[i][j-1]);
      }
    }
    
    // Loop over rows, update inner columns, vectorized
    for (int i = 1; i < worker_height + 1; ++i) {
      // Illustration of two stencils along width
      //      a   b
      //  c   d   e   f
      //      g   h
      
      // Declare float2 pointers
      const float2 * __restrict__ north  = (float2 *) &in[i - 1][0]; // Points to a
      const float2 * __restrict__ middle = (float2 *) &in[i + 0][0]; // Points to d
      const float2 * __restrict__ south  = (float2 *) &in[i + 1][0]; // Points to g
      
      // Loop over inner columns, vectorized
      for (int j = 1; j < half_width; ++j) {
        temp.x = middle[j - 1].y + middle[j].y; // c and e (sides for left stencil)
        temp.y = middle[j].x + middle[j + 1].x; // d and f (sides for right stencil)
        
        // Vectorized computation, reuse temp
        temp = beta*middle[j] + alpha*(north[j] + temp + south[j]);
        
        // Store results
        out[i - 1][2*j - 1] = temp.x; // Output for left stencil
        out[i - 1][2*j - 0] = temp.y; // Output for right stencil
      }
    }
    return true;
  }
};