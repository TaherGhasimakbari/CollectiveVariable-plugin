#ifndef UTIL_AUTOCORRELATION_TPP
#define UTIL_AUTOCORRELATION_TPP

/*
* Simpatico - Simulation Package for Polymeric and Molecular Liquids
*
* Copyright 2010 - 2014, The Regents of the University of Minnesota
* Distributed under the terms of the GNU General Public License.
*/

#include "AutoCorrelation.h"  
#include "AutoCorrStage.tpp"  

#include <string>

namespace Util
{

   /*
   * Constructor
   */
   template <typename Data, typename Product>
   AutoCorrelation<Data, Product>::AutoCorrelation()
    : AutoCorrStage<Data, Product>()
   {
      descendants_.append(this);
   }

   /*
   * Return the maximum delay.
   */
   template <typename Data, typename Product> 
   int AutoCorrelation<Data, Product>::maxDelay() const
   {
      int iStage = descendants_.size() - 1;
      AutoCorrStage<Data, Product>* stagePtr = descendants_[iStage];
      int interval = stagePtr->stageInterval();
      int size = stagePtr->bufferSize();
      return (size - 1)*interval;
   }

   /*
   * Register the creation of a descendant stage.
   */
   template <typename Data, typename Product> 
   void 
   AutoCorrelation<Data, Product>
                  ::registerDescendant(AutoCorrStage<Data, Product>* ptr)
   {  descendants_.append(ptr); }

}
#endif
