#include "DArray.h"
#include "GArray.h"
#include "RingBuffer.h"
#include "AutoCorrStage.tpp"
#include "AutoCorrelation.tpp"
#include <complex>

int main(int argc, char **argv)
{
   typedef std::complex<double> Complex;

   Util::DArray<Complex> darray;
   Util::GArray<Complex> garray;
   Util::RingBuffer<Complex> ring;
   Util::AutoCorrStage<Complex, Complex> stage; 
   Util::AutoCorrelation<Complex, Complex> accumulator; 

   darray.allocate(50);
   ring.allocate(50);
   accumulator.setParam(64, 0, 2);

   return 0;
}
