#include <poplar/Vertex.hpp>
#include <ipudef.h>

using namespace poplar;

class TriadVertex : public Vertex {
public:
  TriadVertex();

  Output<Vector<float>> a;
  Input<Vector<float>> b;
  Input<Vector<float>> c;
  const float q;

  bool compute () {
    for (std::size_t i = 0; i < a.size(); ++i)
      a[i] = b[i] + q*c[i];
    
    return true;
  }
};

class [[ poplar::constraint("elem(*b)!=elem(*c)") ]]
  TriadVertexMemory : public Vertex {
public:
  TriadVertexMemory();

  Output<Vector<float, VectorLayout::SPAN, 8, false>> a;
  Input<Vector<float, VectorLayout::SPAN, 8, true>> b;
  Input<Vector<float, VectorLayout::SPAN, 8, true>> c;
  const float q;

  bool compute () {
    for (std::size_t i = 0; i < a.size(); ++i)
      a[i] = b[i] + q*c[i];
    
    return true;
  }
};

class [[ poplar::constraint("elem(*b)!=elem(*c)") ]]
  TriadVertexAssembly : public Vertex {
public:
  TriadVertexAssembly();

  Output<Vector<float, VectorLayout::SPAN, 8, false>> a;
  Input<Vector<float, VectorLayout::SPAN, 8, true>> b;
  Input<Vector<float, VectorLayout::SPAN, 8, true>> c;
  const float q;

  bool compute () {
    const std::size_t iter = a.size()/2 - 2; 
    float2 tmp; // Temporary float2 variable
    uint2 packed_ptr = __builtin_ipu_tapack(&c[0], &b[0], &a[0]);

    // NOTES:
    // "f32v2axpy" does a*x + y (for 2x 32-bit elements)
    // Prior to using "f32v2axpy", the scalar (a) must be loaded to the $TAS internal state element
    // The current results will be stored in the accumulator state
    // The previous results will be stored to destination 
    __asm__ volatile(
      R"(
      {
        ld2x64pace $a0:1, $a2:3, %[ptr]+=, $m15, 0
        uput $TAS, %[q]
      }
      {
        ld2x64pace $a0:1, $a2:3, %[ptr]+=, $m15, 0
        f32v2axpy %[tmp], $a0:1, $a2:3
      }
      {
        ld2x64pace $a0:1, $a2:3, %[ptr]+=, $m15, 0
        f32v2axpy %[tmp], $a0:1, $a2:3
      }
      nop
      rpt %[iter], (2f - 1f)/8 - 1 
      1:
      {
        ld2xst64pace $a0:3, %[tmp], %[ptr]+=, $m15, 0
        f32v2axpy %[tmp], $a0:1, $a2:3
      }
      2:
      {
        st64pace %[tmp], %[ptr]+=, $m15, 0
        f32v2gina %[tmp], $a14:15, 0
      }
      st64pace %[tmp], %[ptr]+=, $m15, 0
      )"
      // Write List
      : [tmp] "=r"(tmp)
      // Read List
      : [ptr] "r"(packed_ptr), [iter] "r"(iter), [q] "r"(q)
      // Register clobber list
      : "$a0", "$a1", "$a2", "$a3", "memory"
    );

    return true;
  }
};

/*
// Commented Assembly:
// -------------------
// main: load first 64-bits from b and c
// aux: load q to $TAS register
{
  ld2x64pace $a0:1, $a2:3, %[ptr]+=, $m15, 0
  uput $TAS, %[q]
}
// --------------------------------------------
// main: load next 64-bits from b and c
// aux: compute first result to AACC[0, 2] registers
{
  ld2x64pace $a0:1, $a2:3, %[ptr]+=, $m15, 0
  f32v2axpy %[tmp], $a0:1, $a2:3
}
// --------------------------------------------
// main: load next 64-bits from b and c
// aux: store previous result to %[tmp] and compute new results to AACC[0, 2]
{
  ld2x64pace $a0:1, $a2:3, %[ptr]+=, $m15, 0
  f32v2axpy %[tmp], $a0:1, $a2:3
}
// --------------------------------------------
// rpt loop (nop will 8-Byte align the rpt body - between 1: and 2:)
// main: load next elements b[i+1] and c[i+1] and store previous computation a[i-1]
// aux: compute b[i] + q*c[i] 
nop
rpt %[iter], (2f - 1f)/8 - 1 
1:
{
  ld2xst64pace $a0:3, %[tmp], %[ptr]+=, $m15, 0
  f32v2axpy %[tmp], $a0:1, $a2:3
}
2:
// --------------------------------------------
// main: store second-last result from %[tmp] to memory
// aux: store last result from AACC[0, 2] to %[tmp]
{
  st64pace %[tmp], %[ptr]+=, $m15, 0
  f32v2gina %[tmp], $a14:15, 0
}
// --------------------------------------------
// main: store last result from %[tmp] to memory
st64pace %[tmp], %[ptr]+=, $m15, 0
*/