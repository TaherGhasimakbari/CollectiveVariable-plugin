/*
* Simpatico - Simulation Package for Polymeric and Molecular Liquids
*
* Copyright 2010 - 2014, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*/

#include "/home/morsedc/tghasi/CollectiveVariable-plugin/util/Average.h"         // class header
#include <math.h>

namespace Util
{

   /*
   * Default constructor.
   */
   Average::Average(int blockFactor)
    : AverageStage(blockFactor),
      ParamComposite(),
      descendants_(),
      blockSum_(0.0),
      iBlock_(0),
      nSamplePerBlock_(0)
   {
      setClassName("Average");
      // Register self as first "descendant" AverageStage.
      descendants_.push_back(this);
   }

   /*
   * Destructor.
   */
   Average::~Average()
   {}

   /*
   * Reset all accumulators and counters to zero.
   */
   void Average::clear()
   {
      blockSum_ = 0;
      iBlock_   = 0;
      AverageStage::clear();
   }

   /*
   * Set nSamplePerBlock parameter.
   */
   void Average::setNSamplePerBlock(int nSamplePerBlock)
   {  
      if (nSamplePerBlock < 0) {
         UTIL_THROW("Attempt to set nSamplePerBlock < 0");
      }
      nSamplePerBlock_ = nSamplePerBlock; 
   }

   /*
   * Add a sampled value to the ensemble.
   */
   void Average::sample(double value)
   {
      AverageStage::sample(value);

      // Increment block average
      if (nSamplePerBlock_) {
         if (iBlock_ == nSamplePerBlock_) {
            blockSum_ = 0.0;
            iBlock_  = 0;
         }
         blockSum_ += value;
         ++iBlock_;
      }
   }

   /*
   * Add a sampled value and output block average if complete.
   */
   void Average::sample(double value, std::ostream& out)
   {
      AverageStage::sample(value);

      // Increment block average for output
      if (nSamplePerBlock_) {
         if (iBlock_ == nSamplePerBlock_) {
            blockSum_ = 0.0;
            iBlock_  = 0;
         }
         blockSum_ += value;
         ++iBlock_;
         if (iBlock_ == nSamplePerBlock_) {
            out << Dbl(blockSum_/double(iBlock_)) << "\n";
         }
      }
   }

   /*
   * Return estimate of error on average from blocking analysis.
   */
   double Average::blockingError() const
   {
      // Find first stage (descending) with nSample >= 16
      AverageStage* ptr = 0;
      int n = descendants_.size();
      int i = n;
      int nSample = 1;
      while (nSample < 16 && i > 0) {
         --i;
         ptr = descendants_[i];
         nSample = ptr->nSample();
      }

      double error  = ptr->error();
      double sigma  = error/sqrt(2.0*double(nSample-1));
      double weight = 1.0/(sigma*sigma);
      double sum    = error*weight;
      double norm   = weight;
      double aveErr = error;
      double oldSig;

      // Find weighted average within plateau
      bool next = true;
      while (next && i > 0) {
         oldSig = sigma;
         --i;
         ptr = descendants_[i];
         error = ptr->error();
         if (fabs(error - aveErr) < 2.0*oldSig) {
            nSample = ptr->nSample();
            sigma  = error/sqrt(2.0*double(nSample-1));
            weight = 1.0/(sigma*sigma);
            sum   += error*weight;
            norm  += weight;
            aveErr = sum/norm;
         } else {
            next = false;
         }
      }
      return aveErr;
   }

   /*
   * Output statistical properties to file
   */
   void Average::output(std::ostream& out) const
   {
      double aveErr = blockingError();

      out <<  "Average   " << std::setw(20) << average()        
          <<  "  +- "      << std::setw(20) << aveErr << "\n";
      out <<  "Variance  " << std::setw(20) << variance() << "\n";
      out <<  "Std Dev   " << std::setw(20) << stdDeviation() << "\n";
      out <<  "\n";

      out << "Hierarchichal Error Analysis:" << "\n";
      AverageStage* ptr = 0;
      double error;
      int interval;
      int nSample;
      int n = descendants_.size();
      for (int i = 0; i < n; ++i) {
         ptr = descendants_[i];
         error    = ptr->error();
         nSample  = ptr->nSample();
         interval = ptr->stageInterval();
         if (nSample >= 16) {
            out << std::setw(20) << i 
                << std::setw(20) << interval 
                << std::setw(20) << error 
                << std::setw(20) << error/sqrt(double(nSample))
                << std::setw(20) << nSample << "\n";
         }
      }
      out << "\n";
   }

   /*
   * Append pointer to a descendant to descendants_ array.
   */
   void Average::registerDescendant(AverageStage* descendantPtr)
   {  descendants_.push_back(descendantPtr); }

}
