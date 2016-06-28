/*! \file CollectiveVariableGPU.cc
 *  \brief Implements the CollectiveVariableGPU class
 */
#include "CollectiveVariableGPU.h"

#ifdef ENABLE_CUDA

#include "CollectiveVariableGPU.cuh"

CollectiveVariableGPU::CollectiveVariableGPU(boost::shared_ptr<SystemDefinition> sysdef,
                          const std::vector<Scalar>& mode,
                          const std::vector<int3>& lattice_vectors,
                          const std::vector<Scalar>& phases,
                          const std::vector<unsigned int>& grid,
                          const unsigned int& endtime,
                          const std::string& filename,
                          bool overwrite)
    : CollectiveVariable(sysdef, mode, lattice_vectors, endtime, filename, overwrite)
    {

    GPUArray<Scalar> gpu_mode(mode.size(), m_exec_conf);
    m_gpu_mode.swap(gpu_mode);

    // Load mode information
    ArrayHandle<Scalar> h_gpu_mode(m_gpu_mode, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i < mode.size(); i++)
        h_gpu_mode.data[i] = mode[i];

    m_block_size = 512;
    unsigned int max_n_blocks = m_pdata->getMaxN()/m_block_size + 1;

    GPUArray<Scalar2> fourier_mode_scratch(mode.size()*max_n_blocks, m_exec_conf);
    m_fourier_mode_scratch.swap(fourier_mode_scratch);

    m_wave_vectors_updated = 0;
    }

void CollectiveVariableGPU::computeFourierModes(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push(m_exec_conf, "Collective Variable");

    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(), access_location::device, access_mode::read);

    if (m_fourier_mode_scratch.getNumElements() != m_pdata->getMaxN())
        {
        unsigned int max_n_blocks = m_pdata->getMaxN()/m_block_size + 1;
        m_fourier_mode_scratch.resize(max_n_blocks*m_fourier_modes.getNumElements());
        }

        {
        ArrayHandle<Scalar3> d_wave_vectors(m_wave_vectors, access_location::device, access_mode::read);
        ArrayHandle<Scalar2> d_fourier_modes(m_fourier_modes, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_gpu_mode(m_gpu_mode, access_location::device, access_mode::read);
        ArrayHandle<Scalar2> d_fourier_mode_scratch(m_fourier_mode_scratch, access_location::device, access_mode::overwrite);

        // calculate Fourier modes
        gpu_calculate_fourier_modes(m_wave_vectors.getNumElements(),
                                    d_wave_vectors.data,
                                    m_pdata->getN(),
                                    d_postype.data,
                                    d_gpu_mode.data,
                                    d_fourier_modes.data,
                                    m_block_size,
                                    d_fourier_mode_scratch.data
                                    );

        CHECK_CUDA_ERROR();
        }

    if (m_prof)
        m_prof->pop(m_exec_conf);
    }


void export_CollectiveVariableGPU()
    {
    class_<CollectiveVariableGPU, boost::shared_ptr<CollectiveVariableGPU>, bases<CollectiveVariable>, boost::noncopyable >
        ("CollectiveVariableGPU", init< boost::shared_ptr<SystemDefinition>,
                                         const std::vector<Scalar>&,
                                         const std::vector<int3>&,
                                         const std::vector<Scalar>&,
                                         const std::vector<unsigned int>&,
                                         const unsigned int&,
                                         const std::string&,
                                         bool>());

    class_<std::vector<Scalar> >("std_vector_scalar")
        .def(vector_indexing_suite< std::vector<Scalar> > ())
        ;

    class_<std::vector<unsigned int> >("std_vector_uint")
        .def(vector_indexing_suite< std::vector<unsigned int> > ())
        ;

    class_<std::vector<int3> >("std_vector_int3")
        .def(vector_indexing_suite< std::vector<int3> > ())
        ;
    }
#endif
