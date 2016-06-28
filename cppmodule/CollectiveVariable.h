#ifndef __COLLECTIVE_VARIABLE_H__
#define __COLLECTIVE_VARIABLE_H__

/*! \file CollectiveVariable.h
    \brief Declares the CollectiveVariable class
 */
#include <hoomd/hoomd.h>

//#include </home/morsedc/tghasi/CollectiveVariable-plugin/cppmodule/Average.h>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <complex>

// need to declare these classes with __host__ __device__ qualifiers when building in nvcc
// HOiSTDEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

typedef std::complex<float> Complex;
//using namespace Util;

//! Collective variable for studying phase transitions in block copolymer systems
class CollectiveVariable : public Analyzer
    {
    public:
        /*! Constructs the structure factor plugin
            \param sysdef The system definition
            \param mode The per-type coefficients of the Fourier mode
            \param lattice_vectors The Miller indices of the mode vector
            \param filename Filename for the Van-Hove results
            \param overwrite Whether the log file should be overwritten
         */
        CollectiveVariable(boost::shared_ptr<SystemDefinition> sysdef,
                               const std::vector<Scalar>& mode,
                               const std::vector<int3>& lattice_vectors,
                               const std::vector<Scalar>& phases,
                               const std::vector<unsigned int>& grid,
                               const unsigned int& endtime,
                               const std::string& filename,
                               bool overwrite=false
                               );
        virtual ~CollectiveVariable() {}

        void analyze(unsigned int timestep);

    protected:
        std::string m_filename;               //!< The name of log file for CV
        std::ofstream m_file;                 //!< The VanHove file handle
        std::string m_delimiter;              //!< Record delimiter
        bool m_appending;                     //!< Whether we are appending to the log file
        std::vector<int3> m_lattice_vectors;  //!< Stores the list of miller indices
        std::vector<Scalar> m_phases;         //!< Stores the list of miller indices
        std::vector<unsigned int> m_grid;     //!< Stores the list of miller indices
        unsigned int m_endtime;               //!< Stores the last step of VanHove
        std::vector<Scalar> m_mode;           //!< Stores the per-type mode coefficients

        GPUArray<Scalar3> m_wave_vectors;     //!< GPUArray of wave vectors
        GPUArray<Scalar2> m_fourier_modes;    //!< Fourier modes
/*
        Util::Average m_accumulator0;         //!< The accumulator that keeps CVmax data
        Util::Average m_accumulator1;         //!< The accumulator that keeps CVmax/A1 data
        Util::Average m_accumulator2;         //!< The accumulator that keeps CVmax/Astd data
        Util::Average m_accumulator3;         //!< The accumulator that keeps CVmax/A1/Astd data
        Util::Average m_accumulator4;         //!< The accumulator that keeps Astd data
*/

        //! Helper function to update the wave vectors
        void calculateWaveVectors();

        //! Calculates the current value of the collective variable
        virtual void computeFourierModes(unsigned int timestep);

    private:
        void openOutputFiles();

    };

// ------------ Vector math functions --------------------------
//! Comparison operator needed for export of std::vector<int3>
HOSTDEVICE inline bool operator== (const int3 &a, const int3 &b)
    {
    return (a.x == b.x &&
            a.y == b.y &&
            a.z == b.z);
    }

// ------------ Vector math functions --------------------------
//! Comparison operator needed for export of std::vector<int3>
HOSTDEVICE inline Scalar sdot (const Scalar3 &a, const Scalar3 &b)
    {
    return (a.x*b.x +
            a.y*b.y +
            a.z*b.z);
    }

//! Export CollectiveVariable to python
void export_CollectiveVariable();

#endif // __STRUCTURE_FACTOR_H__
