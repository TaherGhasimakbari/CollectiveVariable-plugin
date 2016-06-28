/* \file CollectiveVariable.cc
 * \brief Implements the CollectiveVariable class
 */
#include "CollectiveVariable.h"

#include <iomanip>
#include <stdexcept>
#include <complex>
#include <cmath>

#include <boost/python.hpp>
#include <boost/filesystem.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace boost::python;
using namespace boost::filesystem;
using namespace std;

CollectiveVariable::CollectiveVariable(boost::shared_ptr<SystemDefinition> sysdef,
                               const std::vector<Scalar>& mode,
                               const std::vector<int3>& lattice_vectors,
                               const std::vector<Scalar>& phases,
                               const std::vector<unsigned int>& grid,
                               const unsigned int& endtime,
                               const std::string& filename,
                               bool overwrite)
    : Analyzer(sysdef), m_mode(mode), m_lattice_vectors(lattice_vectors), m_phases(phases), m_grid(grid)
    {
    if (mode.size() != m_pdata->getNTypes())
        {
        m_exec_conf->msg->error() << "cv.lamellar: Number of mode parameters has to equal the number of particle types!" << std::endl;
        throw runtime_error("Error initializing cv.lamellar");
        }

    // allocate array of wave vectors
    GPUArray<Scalar3> wave_vectors(m_lattice_vectors.size(), m_exec_conf);
    m_wave_vectors.swap(wave_vectors);

    GPUArray<Scalar2> fourier_modes(m_lattice_vectors.size(), m_exec_conf);
    m_fourier_modes.swap(fourier_modes);
/*
    m_accumulator0.clear();
    m_accumulator1.clear();
    m_accumulator2.clear();
    m_accumulator3.clear();
    m_accumulator4.clear();
*/
    m_endtime = endtime;
    m_filename = filename;
    m_delimiter = "\t";
    m_appending = !overwrite;
    openOutputFiles();
    }

void CollectiveVariable::openOutputFiles()
    {
#ifdef ENABLE_MPI
    // only output to file on root processor
    if (m_pdata->getDomainDecomposition())
        if (! m_exec_conf->isRoot())
            return;
#endif
    // open the file
    if (exists(m_filename) && m_appending)
        {
        m_exec_conf->msg->notice(3) << "CollectiveVariable-plugin.analyze: Appending log to existing file \"" << m_filename << "\"" << endl;
        m_file.open(m_filename.c_str(), ios_base::in | ios_base::out | ios_base::ate);
        }
    else
        {
        m_exec_conf->msg->notice(3) << "CollectiveVariable-plugin.analzye: Creating new log in file \"" << m_filename << "\"" << endl;
        m_file.open(m_filename.c_str(), ios_base::out);
        m_appending = false;
        }

    if (!m_file.good())
        {
        m_exec_conf->msg->error() << "CollectiveVariable-plugin.analzye: Error opening log file " << m_filename << endl;
        throw runtime_error("Error initializing Logger");
        }
    }

void CollectiveVariable::analyze(unsigned int timestep)
    {
    calculateWaveVectors();

    this->computeFourierModes(timestep);

    ArrayHandle<Scalar2> h_fourier_modes(m_fourier_modes, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_wave_vectors(m_wave_vectors, access_location::host, access_mode::read);

    const BoxDim &box = m_pdata->getGlobalBox();
    const Scalar3 L = box.getL();	    
    const Scalar V = L.x*L.y*L.z;

    bool is_root = true;
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        is_root = m_exec_conf->isRoot();
#endif

    int nWave = m_lattice_vectors.size();
    Scalar re[nWave];    
    Scalar im[nWave];    

    // Calculate value of collective variable (sum of real parts of fourier modes)
    for (unsigned k = 0; k < nWave; k++)
        {
        //Scalar re[k] = Scalar(0.0);
        //Scalar im[k] = Scalar(0.0);

#ifdef ENABLE_MPI
        // reduce value of fourier mode on root processor
        if (m_pdata->getDomainDecomposition())
            {
            MPI_Reduce(&h_fourier_modes.data[k].x,&re[k],1, MPI_HOOMD_SCALAR, MPI_SUM, 0, m_exec_conf->getMPICommunicator());
            MPI_Reduce(&h_fourier_modes.data[k].y,&im[k],1, MPI_HOOMD_SCALAR, MPI_SUM, 0, m_exec_conf->getMPICommunicator());
            }
        else
#endif
            {
            re[k] = h_fourier_modes.data[k].x/sqrt(V);
            im[k] = h_fourier_modes.data[k].y/sqrt(V);
            }
        } 

    if (is_root)
        {
        Complex I(0.0,1.0);
        Complex Amp(0.0,0.0);
        Complex CV(0.0,0.0);
        float CVmax=0.0;
        Scalar3 q;
        Scalar3 r;

        float A1 = 0.0;
        float A2 = 0.0;
        float Astd = 0.0;

        for (int w=0; w<nWave; w++)
            {
            A1 += std::sqrt(re[w]*re[w] + im[w]*im[w]);
            A2 += re[w]*re[w] + im[w]*im[w];
            }
            A1 /= nWave;
            A2 /= nWave;
            Astd = std::sqrt(A2 - A1*A1);
            Astd /= A1;

            int imax1=0;
            int jmax1=0;
            int kmax1=0;
            for (int i=0; i<=m_grid[0]; i++) {
                for (int j=0; j<=m_grid[1]; j++) {
                    for (int k=0; k<=m_grid[2]; k++) {
                        CV = Complex(0.0,0.0);
                        r = make_scalar3(i*L.x/m_grid[0],j*L.y/m_grid[1],k*L.z/m_grid[2]);
                        for (int w=0; w<nWave; w++) {
                        Amp = Complex(re[w], im[w]);
                        q = h_wave_vectors.data[w];
                        CV += exp(-I*static_cast<Complex>(m_phases[w]+sdot(q,r)))*Amp;
                        }
                        if (CV.real() > CVmax) {
                        CVmax = CV.real()/nWave;
                        imax1 = i;
                        jmax1 = j;
                        kmax1 = k;
                        }
                    }
                }
            }
/*
                        CV = Complex(0.0,0.0);
                        r = make_scalar3(imax1*L.x/m_grid[0],jmax1*L.y/m_grid[1],kmax1*L.z/m_grid[2]);
                        std::cout<< imax1*L.x/m_grid[0] << " " << jmax1*L.y/m_grid[1] << " " << kmax1*L.z/m_grid[2] << " " <<"\n\n";
                        for (int w=0; w<nWave; w++) {
                        Amp = Complex(re[w], im[w]);
                        std::cout<< L.x << " " << L.y << " " << L.z << " " <<"\t\t";
                        std::cout<< Complex(re[w], im[w])<<"\t\t";
                        q = h_wave_vectors.data[w];
                        std::cout<< q.x << " " << q.y << " " << q.z << " " <<"\t\t";
                        CV += exp(-I*static_cast<Complex>(m_phases[w]+sdot(q,r)))*Amp;
                        std::cout << sdot(q,r) << "\t\t";
                        std::cout<< exp(-I*static_cast<Complex>(m_phases[w]+sdot(q,r)))*Amp <<"\n";
                        }
                        std::cout<< "CVmax/A1"<<CVmax/A1<<"\n\n";
*/
            int imax2=0;
            int jmax2=0;
            int kmax2=0;
            Scalar3 rmax1;
            rmax1 = make_scalar3(imax1*L.x/m_grid[0],jmax1*L.y/m_grid[1],kmax1*L.z/m_grid[2]);
            for (int i=-m_grid[0]; i<m_grid[0]; i++) {
                for (int j=-m_grid[1]; j<m_grid[1]; j++) {
                    for (int k=-m_grid[2]; k<m_grid[2]; k++) {
                        CV = Complex(0.0,0.0);
                        r = rmax1;
                        r += make_scalar3(i*L.x/m_grid[0]/m_grid[0],j*L.y/m_grid[1]/m_grid[1],k*L.z/m_grid[2]/m_grid[2]);
                        for (int w=0; w<nWave; w++) {
                        Amp = Complex(re[w], im[w]);
                        q = h_wave_vectors.data[w];
                        CV += exp(-I*static_cast<Complex>(m_phases[w]+sdot(q,r)))*Amp;
                        }
                        if (CV.real() > CVmax) {
                        CVmax = CV.real()/nWave;
                        imax2 = i;
                        jmax2 = j;
                        kmax2 = k;
                        }
                    }
                }
            }

            Scalar3 rmax2;
            rmax2 = rmax1;
            rmax2 += make_scalar3(imax2*L.x/m_grid[0]/m_grid[0],jmax2*L.y/m_grid[1]/m_grid[1],kmax2*L.z/m_grid[2]/m_grid[2]);
            for (int i=-m_grid[0]; i<m_grid[0]; i++) {
                for (int j=-m_grid[1]; j<m_grid[1]; j++) {
                    for (int k=-m_grid[2]; k<m_grid[2]; k++) {
                        CV = Complex(0.0,0.0);
                        r = rmax2;
                        r += make_scalar3(i*L.x/m_grid[0]/m_grid[0]/m_grid[0],j*L.y/m_grid[1]/m_grid[1]/m_grid[1],k*L.z/m_grid[2]/m_grid[2]/m_grid[2]);
                        for (int w=0; w<nWave; w++){
                        Amp = Complex(re[w], im[w]);
                        q = h_wave_vectors.data[w];
                        CV += exp(-I*static_cast<Complex>(m_phases[w]+sdot(q,r)))*Amp;
                        }
                        if (CV.real() > CVmax){
                        CVmax = CV.real()/nWave;
                        }
                    }
                }
            }

            m_file << setprecision(10) << timestep << m_delimiter;
            m_file << setprecision(10) << A1 << m_delimiter;
            m_file << setprecision(10) << CVmax << m_delimiter;
            m_file << setprecision(10) << CVmax/A1 << m_delimiter;
            m_file << setprecision(10) << CVmax/A1/Astd << m_delimiter;
            m_file << setprecision(10) << Astd << m_delimiter <<"\n";

/*
            m_accumulator0.sample(CVmax);
            m_accumulator1.sample(CVmax/A1);
            m_accumulator2.sample(CVmax/Astd);
            m_accumulator3.sample(CVmax/A1/Astd);
            m_accumulator4.sample(Astd);
*/
 
/*            // Calculate value of collective variable (sum of real parts of fourier modes)
            if (timestep == m_endtime) {
                m_exec_conf->msg->notice(3) << "CVmax: Creating new log in file: " << m_filename << endl;
                std::stringstream ss0;
                ss0 << m_filename;
                ss0 << "max";
                m_file.open(ss0.str().c_str(), ios_base::out);
                m_accumulator0.output(m_file);
                m_file.close();
                if (!m_file.good()) {
                    m_exec_conf->msg->error() << "CVmax: Error opening file: " << m_filename << endl;
                    throw runtime_error("Error Initializing VanHove Files");
                }

                m_exec_conf->msg->notice(3) << "CVmaxA1: Creating new log in file: " << m_filename << endl;
                std::stringstream ss1;
                ss1 << m_filename;
                ss1 << "maxA1";
                m_file.open(ss1.str().c_str(), ios_base::out);
                m_accumulator1.output(m_file);
                m_file.close();
                if (!m_file.good()) {
                    m_exec_conf->msg->error() << "CVmaxA1: Error opening file: " << m_filename << endl;
                    throw runtime_error("Error Initializing VanHove Files");
                }

                m_exec_conf->msg->notice(3) << "CVmaxAstd: Creating new log in file: " << m_filename << endl;
                std::stringstream ss2;
                ss2 << m_filename;
                ss2 << "maxAstd";
                m_file.open(ss2.str().c_str(), ios_base::out);
                m_accumulator2.output(m_file);
                m_file.close();
                if (!m_file.good()) {
                    m_exec_conf->msg->error() << "CVmaxAstd: Error opening file: " << m_filename << endl;
                    throw runtime_error("Error Initializing VanHove Files");
                }

                m_exec_conf->msg->notice(3) << "CVmaxA1Astd: Creating new log in file: " << m_filename << endl;
                std::stringstream ss3;
                ss3 << m_filename;
                ss3 << "maxA1Astd";
                m_file.open(ss3.str().c_str(), ios_base::out);
                m_accumulator3.output(m_file);
                m_file.close();
                if (!m_file.good()) {
                    m_exec_conf->msg->error() << "CVmaxA1Astd: Error opening file: " << m_filename << endl;
                    throw runtime_error("Error Initializing VanHove Files");
                }

                m_exec_conf->msg->notice(3) << "Astd: Creating new log in file: " << m_filename << endl;
                std::stringstream ss4;
                ss4 << m_filename;
                ss4 << "Astd";
                m_file.open(ss4.str().c_str(), ios_base::out);
                m_accumulator4.output(m_file);
                m_file.close();
                if (!m_file.good()) {
                    m_exec_conf->msg->error() << "Astd: Error opening file: " << m_filename << endl;
                    throw runtime_error("Error Initializing VanHove Files");
                }
            }
*/
    }
    }


//! Calculate wave vectors
void CollectiveVariable::calculateWaveVectors()
    {
    ArrayHandle<Scalar3> h_wave_vectors(m_wave_vectors, access_location::host, access_mode::overwrite);

    const BoxDim &box = m_pdata->getGlobalBox();
    const Scalar3 L = box.getL();

    for (unsigned int k = 0; k < m_lattice_vectors.size(); k++)
        h_wave_vectors.data[k] = 2*M_PI*make_scalar3(m_lattice_vectors[k].x/L.x,
                                              m_lattice_vectors[k].y/L.y,
                                              m_lattice_vectors[k].z/L.z);
    }

//! Returns a list of fourier modes (for all wave vectors)
void CollectiveVariable::computeFourierModes(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push("Structure Factor");

    ArrayHandle<Scalar2> h_fourier_modes(m_fourier_modes, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar3> h_wave_vectors(m_wave_vectors, access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);

    for (unsigned int k = 0; k < m_wave_vectors.getNumElements(); k++)
        {
        h_fourier_modes.data[k] = make_scalar2(0.0,0.0);
        Scalar3 q = h_wave_vectors.data[k];
        
        for (unsigned int idx = 0; idx < m_pdata->getN(); idx++)
            {
            Scalar4 postype = h_postype.data[idx];

            Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
            unsigned int type = __scalar_as_int(postype.w);
            Scalar mode = m_mode[type];
            Scalar dotproduct = sdot(q,pos);
            h_fourier_modes.data[k].x += mode * cos(dotproduct);
            h_fourier_modes.data[k].y += mode * sin(dotproduct);
            }
        }

    if (m_prof) m_prof->pop();
    }

void export_CollectiveVariable()
    {
    class_<CollectiveVariable, boost::shared_ptr<CollectiveVariable>, bases<Analyzer>, boost::noncopyable >
        ("CollectiveVariable", init< boost::shared_ptr<SystemDefinition>,
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
