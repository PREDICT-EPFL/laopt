#ifndef LAOPT_HPIPM_INTERFACE_HPP
#define LAOPT_HPIPM_INTERFACE_HPP

#include "laopt/utility.hpp"
#include "laopt/solvers/qp_base.hpp"
#include "laopt/solvers/sqp_solver.hpp"
#include "laopt/tools/multiple_shooting.hpp"

#include "hpipm_d_ocp_qp.h"
#include "hpipm_d_ocp_qp_ipm.h"
#include "hpipm_d_ocp_qp_dim.h"
#include "hpipm_d_ocp_qp_sol.h"

namespace laopt
{

class HPIPMSolver : public QPBase<HPIPMSolver, double>
{
public:
    using Base = QPBase<HPIPMSolver, double>;
    using scalar_t = typename Base::scalar_t;

private:
    using constraint_t = typename Base::constraint_t;

    Eigen::VectorX<scalar_t> m_b;

    Eigen::VectorX<scalar_t> m_lbx;
    Eigen::VectorX<scalar_t> m_lbx_mask;
    Eigen::VectorX<scalar_t> m_ubx;
    Eigen::VectorX<scalar_t> m_ubx_mask;
    Eigen::VectorX<int> m_idxbx;
    Eigen::VectorX<int> m_idxbxe;

    Eigen::VectorX<scalar_t> m_lbu;
    Eigen::VectorX<scalar_t> m_lbu_mask;
    Eigen::VectorX<scalar_t> m_ubu;
    Eigen::VectorX<scalar_t> m_ubu_mask;
    Eigen::VectorX<int> m_idxbu;
    Eigen::VectorX<int> m_idxbue;

    Eigen::VectorX<scalar_t> m_lg;
    Eigen::VectorX<scalar_t> m_lg_mask;
    Eigen::VectorX<scalar_t> m_ug;
    Eigen::VectorX<scalar_t> m_ug_mask;
    Eigen::VectorX<int> m_idxg; // not needed by HPIPM, but used as mapping
    Eigen::VectorX<int> m_idxge;

    int m_N{0}, m_nx{0}, m_nu{0}, m_ng{0}, m_ng0{0}, m_ngf{0};
    d_ocp_qp_dim m_dim{};
    std::unique_ptr<char[]> m_dim_memory;
    d_ocp_qp m_qp{};
    std::unique_ptr<char[]> m_qp_memory;
    d_ocp_qp_sol m_qp_sol{};
    std::unique_ptr<char[]> m_qp_sol_memory;
    d_ocp_qp_ipm_arg m_qp_ipm_arg{};
    std::unique_ptr<char[]> m_qp_ipm_arg_memory;
    d_ocp_qp_ipm_ws m_qp_ipm_ws{};
    std::unique_ptr<char[]> m_qp_ipm_ws_memory;

public:
    HPIPMSolver(int n, int m) :
        Base(n, m)
    {
        this->m_settings.max_iter = 50;
    }

    void set_problem_dims(int N, int nx, int nu, int ng, int ng0, int ngf)
    {
        m_N = N;
        m_nx = nx;
        m_nu = nu;
        m_ng = ng;
        m_ng0 = ng0;
        m_ngf = ngf;

        m_b.resize(N * nx);

        m_lbx.resize((N + 1) * nx);
        m_lbx_mask.resize((N + 1) * nx);
        m_ubx.resize((N + 1) * nx);
        m_ubx_mask.resize((N + 1) * nx);
        m_idxbx.resize((N + 1) * nx);
        m_idxbxe.resize((N + 1) * nx);

        m_lbu.resize(N * nu);
        m_lbu_mask.resize(N * nu);
        m_ubu.resize(N * nu);
        m_ubu_mask.resize(N * nu);
        m_idxbu.resize(N * nu);
        m_idxbue.resize(N * nu);

        m_lg.resize(ng0 + (N - 1) * ng + ngf);
        m_lg_mask.resize(ng0 + (N - 1) * ng + ngf);
        m_ug.resize(ng0 + (N - 1) * ng + ngf);
        m_ug_mask.resize(ng0 + (N - 1) * ng + ngf);
        m_idxg.resize(ng0 + (N - 1) * ng + ngf);
        m_idxge.resize(ng0 + (N - 1) * ng + ngf);

        // TODO: Remove NaNs
        m_lbx.setConstant(NAN);
        m_lbx_mask.setConstant(NAN);
        m_ubx.setConstant(NAN);
        m_ubx_mask.setConstant(NAN);
        m_idxbx.setConstant(-1);
        m_idxbxe.setConstant(-1);

        m_lbu.setConstant(NAN);
        m_lbu_mask.setConstant(NAN);
        m_ubu.setConstant(NAN);
        m_ubu_mask.setConstant(NAN);
        m_idxbu.setConstant(-1);
        m_idxbue.setConstant(-1);

        m_lg.setConstant(NAN);
        m_lg_mask.setConstant(NAN);
        m_ug.setConstant(NAN);
        m_ug_mask.setConstant(NAN);
        m_idxg.setConstant(-1);
        m_idxge.setConstant(-1);

        hpipm_size_t dim_size = d_ocp_qp_dim_memsize(N);
        m_dim_memory = std::unique_ptr<char[]>(new char[dim_size]);
        d_ocp_qp_dim_create(N, &m_dim, m_dim_memory.get());

        for (int i = 0; i <= N; i++) {
            d_ocp_qp_dim_set_nx(i, nx, &m_dim);
            d_ocp_qp_dim_set_nu(i, nu, &m_dim);
        }
    }

    qp_solver_info_t solve_impl(const Eigen::BlockSparseMatrix<scalar_t>& H,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& f,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xlb,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xub,
                                const Eigen::BlockSparseMatrix<scalar_t>& A,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Alb,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Aub) noexcept
    {
//        std::cout << "H: " << H << std::endl;
//        std::cout << "A: " << A << std::endl;
//        std::cout << "f: " << f.transpose() << std::endl;
//        std::cout << "Alb: " << Alb.transpose() << std::endl;
//        std::cout << "Aub: " << Aub.transpose() << std::endl;
//        std::cout << "xlb: " << xlb.transpose() << std::endl;
//        std::cout << "xub: " << xub.transpose() << std::endl;

        this->parse_constraints_bounds(xlb, xub, Alb, Aub);

        // TODO: reuse pattern

        // box constraints
        int offset = 0;
        for (int i = 0; i <= m_N; i++) {
            int nbx = 0;
            int nbxe = 0;
            for (int j = 0; j < m_nx; j++) {
                if (m_box_constraint_type[offset + j] != constraint_t::UNBOUNDED_CONSTR) {
                    m_idxbx[i * m_nx + nbx] = j;
                    if (m_box_constraint_type[offset + j] == constraint_t::EQ_CONSTR) {
                        m_idxbxe[i * m_nx + nbxe++] = j;
                        m_lbx_mask[i * m_nx + nbx] = 1.0;
                        m_ubx_mask[i * m_nx + nbx] = 1.0;
                    } else if (m_box_constraint_type[offset + j] == constraint_t::INEQ_CONSTR) {
                        m_lbx_mask[i * m_nx + nbx] = 1.0;
                        m_ubx_mask[i * m_nx + nbx] = 1.0;
                    } else if (m_box_constraint_type[offset + j] == constraint_t::INEQ_LB_ONLY_CONSTR) {
                        m_lbx_mask[i * m_nx + nbx] = 1.0;
                    } else if (m_box_constraint_type[offset + j] == constraint_t::INEQ_UB_ONLY_CONSTR) {
                        m_ubx_mask[i * m_nx + nbx] = 1.0;
                    }
                    nbx++;
                }
            }
            offset += m_nx;
            d_ocp_qp_dim_set_nbx(i, nbx, &m_dim);
            d_ocp_qp_dim_set_nbxe(i, nbxe, &m_dim);

            if (i < m_N) {
                int nbu = 0;
                int nbue = 0;
                for (int j = 0; j < m_nu; j++) {
                    if (m_box_constraint_type[offset + j] != constraint_t::UNBOUNDED_CONSTR) {
                        m_idxbu[i * m_nu + nbu] = j;
                        if (m_box_constraint_type[offset + j] == constraint_t::EQ_CONSTR) {
                            m_idxbue[i * m_nu + nbue++] = j;
                            m_lbu_mask[i * m_nu + nbu] = 1.0;
                            m_ubu_mask[i * m_nu + nbu] = 1.0;
                        } else if (m_box_constraint_type[offset + j] == constraint_t::INEQ_CONSTR) {
                            m_lbu_mask[i * m_nu + nbu] = 1.0;
                            m_ubu_mask[i * m_nu + nbu] = 1.0;
                        } else if (m_box_constraint_type[offset + j] == constraint_t::INEQ_LB_ONLY_CONSTR) {
                            m_lbu_mask[i * m_nu + nbu] = 1.0;
                        } else if (m_box_constraint_type[offset + j] == constraint_t::INEQ_UB_ONLY_CONSTR) {
                            m_ubu_mask[i * m_nu + nbu] = 1.0;
                        }
                        nbu++;
                    }
                }
                offset += m_nu;
                d_ocp_qp_dim_set_nbu(i, nbu, &m_dim);
                d_ocp_qp_dim_set_nbue(i, nbue, &m_dim);
            }
        }

//        // general constraints
//        int global_offset = m_N * m_nx; // skip dynamics constraints
//        int local_offset = 0;
//        int ng = 0;
//        int nge = 0;
//        for (int j = 0; j < m_ng0; j++) {
//            if (m_constraint_type[global_offset + j] != constraint_t::UNBOUNDED_CONSTR) {
//                m_idxg[local_offset + ng] = j;
//                if (m_constraint_type[global_offset + j] == constraint_t::EQ_CONSTR) {
//                    m_idxge[local_offset + nge++] = ng;
//                    m_lg_mask[local_offset + ng] = 1.0;
//                    m_ug_mask[local_offset + ng] = 1.0;
//                } else if (m_constraint_type[global_offset + j] == constraint_t::INEQ_CONSTR) {
//                    m_lg_mask[local_offset + ng] = 1.0;
//                    m_ug_mask[local_offset + ng] = 1.0;
//                } else if (m_constraint_type[global_offset + j] == constraint_t::INEQ_LB_ONLY_CONSTR) {
//                    m_lg_mask[local_offset + ng] = 1.0;
//                } else if (m_constraint_type[global_offset + j] == constraint_t::INEQ_UB_ONLY_CONSTR) {
//                    m_ug_mask[local_offset + ng] = 1.0;
//                }
//                ng++;
//            }
//        }
//        global_offset += m_ng0;
//        local_offset += m_ng0;
//        d_ocp_qp_dim_set_ng(0, ng, &m_dim);
//        d_ocp_qp_dim_set_nge(0, nge, &m_dim);
//
//        for (int i = 1; i < m_N; i++) {
//            ng = 0;
//            nge = 0;
//            for (int j = 0; j < m_ng; j++) {
//                if (m_constraint_type[global_offset + j] != constraint_t::UNBOUNDED_CONSTR) {
//                    m_idxg[local_offset + ng] = j;
//                    if (m_constraint_type[global_offset + j] == constraint_t::EQ_CONSTR) {
//                        m_idxge[local_offset + nge++] = ng;
//                        m_lg_mask[local_offset + ng] = 1.0;
//                        m_ug_mask[local_offset + ng] = 1.0;
//                    } else if (m_constraint_type[global_offset + j] == constraint_t::INEQ_CONSTR) {
//                        m_lg_mask[local_offset + ng] = 1.0;
//                        m_ug_mask[local_offset + ng] = 1.0;
//                    } else if (m_constraint_type[global_offset + j] == constraint_t::INEQ_LB_ONLY_CONSTR) {
//                        m_lg_mask[local_offset + ng] = 1.0;
//                    } else if (m_constraint_type[global_offset + j] == constraint_t::INEQ_UB_ONLY_CONSTR) {
//                        m_ug_mask[local_offset + ng] = 1.0;
//                    }
//                    ng++;
//                }
//            }
//            global_offset += m_ng;
//            local_offset += m_ng;
//            d_ocp_qp_dim_set_ng(i, ng, &m_dim);
//            d_ocp_qp_dim_set_nge(i, nge, &m_dim);
//        }
//
//        ng = 0;
//        nge = 0;
//        for (int j = 0; j < m_ngf; j++) {
//            if (m_constraint_type[global_offset + j] != constraint_t::UNBOUNDED_CONSTR) {
//                m_idxg[local_offset + ng] = j;
//                if (m_constraint_type[global_offset + j] == constraint_t::EQ_CONSTR) {
//                    m_idxge[local_offset + nge++] = ng;
//                    m_lg_mask[local_offset + ng] = 1.0;
//                    m_ug_mask[local_offset + ng] = 1.0;
//                } else if (m_constraint_type[global_offset + j] == constraint_t::INEQ_CONSTR) {
//                    m_lg_mask[local_offset + ng] = 1.0;
//                    m_ug_mask[local_offset + ng] = 1.0;
//                } else if (m_constraint_type[global_offset + j] == constraint_t::INEQ_LB_ONLY_CONSTR) {
//                    m_lg_mask[local_offset + ng] = 1.0;
//                } else if (m_constraint_type[global_offset + j] == constraint_t::INEQ_UB_ONLY_CONSTR) {
//                    m_ug_mask[local_offset + ng] = 1.0;
//                }
//                ng++;
//            }
//        }
//        // global_offset += m_ngf;
//        // local_offset += m_ngf;
//        d_ocp_qp_dim_set_ng(m_N, ng, &m_dim);
//        d_ocp_qp_dim_set_nge(m_N, nge, &m_dim);

        hpipm_size_t qp_size = d_ocp_qp_memsize(&m_dim);
        m_qp_memory = std::unique_ptr<char[]>(new char[qp_size]);
        d_ocp_qp_create(&m_dim, &m_qp, m_qp_memory.get());

        // copy data
        m_b = -Alb.head(m_N * m_nx);
        for (int i = 0; i < m_N; i++) {
            int nbx;
            d_ocp_qp_dim_get_nbx(&m_dim, i, &nbx);
            for (int j = 0; j < nbx; j++) {
                int idxbx = i * (m_nx + m_nu) + m_idxbx[i * m_nx + j];
                m_lbx[i * m_nx + j] = xlb[idxbx];
                m_ubx[i * m_nx + j] = xub[idxbx];
            }

            int nbu;
            d_ocp_qp_dim_get_nbu(&m_dim, i, &nbu);
            for (int j = 0; j < nbu; j++) {
                int idxbu = i * (m_nx + m_nu) + m_nx + m_idxbu[i * m_nu + j];
                m_lbu[i * m_nu + j] = xlb[idxbu];
                m_ubu[i * m_nu + j] = xub[idxbu];
            }
        }
        int nbx;
        d_ocp_qp_dim_get_nbx(&m_dim, m_N, &nbx);
        for (int j = 0; j < nbx; j++) {
            int idxbx = m_N * (m_nx + m_nu) + m_idxbx[m_N * m_nx + j];
            m_lbx[m_N * m_nx + j] = xlb[idxbx];
            m_ubx[m_N * m_nx + j] = xub[idxbx];
        }

//        global_offset = m_N * m_nx;
//        local_offset = 0;
//        d_ocp_qp_dim_get_ng(&m_dim, 0, &ng);
//        for (int j = 0; j < ng; j++) {
//            int idxg = global_offset + m_idxg[j];
//            m_lg[local_offset + j] = Alb[idxg];
//            m_ug[local_offset + j] = Aub[idxg];
//        }
//        global_offset += m_ng0;
//        local_offset += m_ng0;
//        for (int i = 1; i < m_N; i++) {
//            d_ocp_qp_dim_get_ng(&m_dim, i, &ng);
//            for (int j = 0; j < ng; j++) {
//                int idxg = global_offset + m_idxg[j];
//                m_lg[local_offset + j] = Alb[idxg];
//                m_ug[local_offset + j] = Aub[idxg];
//            }
//            global_offset += m_ng;
//            local_offset += m_ng;
//        }
//        d_ocp_qp_dim_get_ng(&m_dim, m_N, &ng);
//        for (int j = 0; j < ng; j++) {
//            int idxg = global_offset + m_idxg[j];
//            m_lg[local_offset + j] = Alb[idxg];
//            m_ug[local_offset + j] = Aub[idxg];
//        }
//        // global_offset += m_ngf;
//        // local_offset += m_ngf;

//        std::cout << "m_lbx: " << m_lbx.transpose() << std::endl;
//        std::cout << "m_lbx_mask: " << m_lbx_mask.transpose() << std::endl;
//        std::cout << "m_ubx: " << m_ubx.transpose() << std::endl;
//        std::cout << "m_ubx_mask: " << m_ubx_mask.transpose() << std::endl;
//        std::cout << "m_idxbx: " << m_idxbx.transpose() << std::endl;
//        std::cout << "m_idxbxe: " << m_idxbxe.transpose() << std::endl;
//
//        std::cout << "m_lbu: " << m_lbu.transpose() << std::endl;
//        std::cout << "m_lbu_mask: " << m_lbu_mask.transpose() << std::endl;
//        std::cout << "m_ubu: " << m_ubu.transpose() << std::endl;
//        std::cout << "m_ubu_mask: " << m_ubu_mask.transpose() << std::endl;
//        std::cout << "m_idxbu: " << m_idxbu.transpose() << std::endl;
//        std::cout << "m_idxbue: " << m_idxbue.transpose() << std::endl;
//
//        std::cout << "m_lg: " << m_lg.transpose() << std::endl;
//        std::cout << "m_lg_mask: " << m_lg_mask.transpose() << std::endl;
//        std::cout << "m_ug: " << m_ug.transpose() << std::endl;
//        std::cout << "m_ug_mask: " << m_ug_mask.transpose() << std::endl;
//        std::cout << "m_idxg: " << m_idxg.transpose() << std::endl;
//        std::cout << "m_idxge: " << m_idxge.transpose() << std::endl;

        // set data
        for (int i = 0; i < m_N; i++) {
            // dynamics
            double* Ai = const_cast<double*>(A.coeff(i, 2 * i).data());
            double* Bi = const_cast<double*>(A.coeff(i, 2 * i + 1).data());
            double* bi = m_b.data() + m_nx * i;
            d_ocp_qp_set_A(i, Ai, &m_qp);
            d_ocp_qp_set_B(i, Bi, &m_qp);
            d_ocp_qp_set_b(i, bi, &m_qp);

            // cost
            double* Qi = const_cast<double*>(H.coeff(2 * i, 2 * i).data());
            double* Si = const_cast<double*>(H.coeff(2 * i, 2 * i + 1).data());
            double* Ri = const_cast<double*>(H.coeff(2 * i + 1, 2 * i + 1).data());
            double* qi = const_cast<double*>(f.data()) + (m_nx + m_nu) * i;
            double* ri = const_cast<double*>(f.data()) + (m_nx + m_nu) * i + m_nx;
            d_ocp_qp_set_Q(i, Qi, &m_qp);
            d_ocp_qp_set_S(i, Si, &m_qp);
            d_ocp_qp_set_R(i, Ri, &m_qp);
            d_ocp_qp_set_q(i, qi, &m_qp);
            d_ocp_qp_set_r(i, ri, &m_qp);

            // box constraints
            int x_offset = i * m_nx;
            d_ocp_qp_set_lbx(i, m_lbx.data() + x_offset, &m_qp);
            d_ocp_qp_set_lbx_mask(i, m_lbx_mask.data() + x_offset, &m_qp);
            d_ocp_qp_set_ubx(i, m_ubx.data() + x_offset, &m_qp);
            d_ocp_qp_set_ubx_mask(i, m_ubx_mask.data() + x_offset, &m_qp);
            d_ocp_qp_set_idxbx(i, m_idxbx.data() + x_offset, &m_qp);
            d_ocp_qp_set_idxbxe(i, m_idxbxe.data() + x_offset, &m_qp);
            int u_offset = i * m_nu;
            d_ocp_qp_set_lbu(i, m_lbu.data() + u_offset, &m_qp);
            d_ocp_qp_set_lbu_mask(i, m_lbu_mask.data() + u_offset, &m_qp);
            d_ocp_qp_set_ubu(i, m_ubu.data() + u_offset, &m_qp);
            d_ocp_qp_set_ubu_mask(i, m_ubu_mask.data() + u_offset, &m_qp);
            d_ocp_qp_set_idxbu(i, m_idxbu.data() + u_offset, &m_qp);
            d_ocp_qp_set_idxbue(i, m_idxbue.data() + u_offset, &m_qp);
        }
        // set final cost
        double* QN = const_cast<double*>(H.coeff(2 * m_N, 2 * m_N).data());
        double* qN = const_cast<double*>(f.data()) + (m_nx + m_nu) * m_N;
        d_ocp_qp_set_Q(m_N, QN, &m_qp);
        d_ocp_qp_set_q(m_N, qN, &m_qp);

        // set final box constraints
        int x_offset = m_N * m_nx;
        d_ocp_qp_set_lbx(m_N, m_lbx.data() + x_offset, &m_qp);
        d_ocp_qp_set_lbx_mask(m_N, m_lbx_mask.data() + x_offset, &m_qp);
        d_ocp_qp_set_ubx(m_N, m_ubx.data() + x_offset, &m_qp);
        d_ocp_qp_set_ubx_mask(m_N, m_ubx_mask.data() + x_offset, &m_qp);
        d_ocp_qp_set_idxbx(m_N, m_idxbx.data() + x_offset, &m_qp);
        d_ocp_qp_set_idxbxe(m_N, m_idxbxe.data() + x_offset, &m_qp);

//        // general constraints
//        global_offset = m_N * m_nx;
//        local_offset = 0;
//        int block_offset = m_N;
//        if (m_ng0 > 0) {
//            double* C0 = const_cast<double*>(A.coeff(block_offset, 0).data());
//            double* D0 = const_cast<double*>(A.coeff(block_offset, 1).data());
//            d_ocp_qp_set_C(0, C0, &m_qp);
//            d_ocp_qp_set_D(0, D0, &m_qp);
//
//            d_ocp_qp_set_lg(0, m_lg.data() + local_offset, &m_qp);
//            d_ocp_qp_set_lg_mask(0, m_lg_mask.data() + local_offset, &m_qp);
//            d_ocp_qp_set_ug(0, m_ug.data() + local_offset, &m_qp);
//            d_ocp_qp_set_ug_mask(0, m_ug_mask.data() + local_offset, &m_qp);
//            d_ocp_qp_set_idxge(0, m_idxge.data() + global_offset, &m_qp);
//
//            global_offset += m_ng0;
//            local_offset += m_ng0;
//            block_offset++;
//        }
//        if (m_ng > 0) {
//            for (int i = 1; i < m_N; i++) {
//                double* Ci = const_cast<double*>(A.coeff(block_offset, 2 * i).data());
//                double* Di = const_cast<double*>(A.coeff(block_offset, 2 * i + 1).data());
//                d_ocp_qp_set_C(i, Ci, &m_qp);
//                d_ocp_qp_set_D(i, Di, &m_qp);
//
//                d_ocp_qp_set_lg(i, m_lg.data() + local_offset, &m_qp);
//                d_ocp_qp_set_lg_mask(i, m_lg_mask.data() + local_offset, &m_qp);
//                d_ocp_qp_set_ug(i, m_ug.data() + local_offset, &m_qp);
//                d_ocp_qp_set_ug_mask(i, m_ug_mask.data() + local_offset, &m_qp);
//                d_ocp_qp_set_idxge(i, m_idxge.data() + global_offset, &m_qp);
//
//                global_offset += m_ng;
//                local_offset += m_ng;
//                block_offset++;
//            }
//        }
//        if (m_ngf > 0) {
//            double* CN = const_cast<double*>(A.coeff(block_offset, 2 * m_N).data());
//            double* DN = const_cast<double*>(A.coeff(block_offset, 2 * m_N + 1).data());
//            d_ocp_qp_set_C(m_N, CN, &m_qp);
//            d_ocp_qp_set_D(m_N, DN, &m_qp);
//
//            d_ocp_qp_set_lg(m_N, m_lg.data() + local_offset, &m_qp);
//            d_ocp_qp_set_lg_mask(m_N, m_lg_mask.data() + local_offset, &m_qp);
//            d_ocp_qp_set_ug(m_N, m_ug.data() + local_offset, &m_qp);
//            d_ocp_qp_set_ug_mask(m_N, m_ug_mask.data() + local_offset, &m_qp);
//            d_ocp_qp_set_idxge(m_N, m_idxge.data() + global_offset, &m_qp);
//
//            // global_offset += m_ngf;
//            // local_offset += m_ngf;
//            // g_offset++;
//        }

        hpipm_size_t qp_sol_size = d_ocp_qp_sol_memsize(&m_dim);
        m_qp_sol_memory = std::unique_ptr<char[]>(new char[qp_sol_size]);
        d_ocp_qp_sol_create(&m_dim, &m_qp_sol, m_qp_sol_memory.get());

        hpipm_size_t qp_ipm_arg_size = d_ocp_qp_ipm_arg_memsize(&m_dim);
        m_qp_ipm_arg_memory = std::unique_ptr<char[]>(new char[qp_ipm_arg_size]);
        d_ocp_qp_ipm_arg_create(&m_dim, &m_qp_ipm_arg, m_qp_ipm_arg_memory.get());
        d_ocp_qp_ipm_arg_set_default(BALANCE, &m_qp_ipm_arg);

        hpipm_size_t qp_ipm_ws_size = d_ocp_qp_ipm_ws_memsize(&m_dim, &m_qp_ipm_arg);
        m_qp_ipm_ws_memory = std::unique_ptr<char[]>(new char[qp_ipm_ws_size]);
        d_ocp_qp_ipm_ws_create(&m_dim, &m_qp_ipm_arg, &m_qp_ipm_ws, m_qp_ipm_ws_memory.get());

        int hpipm_status;
        d_ocp_qp_ipm_solve(&m_qp, &m_qp_sol, &m_qp_ipm_arg, &m_qp_ipm_ws);
        d_ocp_qp_ipm_get_status(&m_qp_ipm_ws, &hpipm_status);

        // extract primal solution
        offset = 0;
        for (int i = 0; i < m_N; i++) {
            d_ocp_qp_sol_get_x(i, &m_qp_sol, this->m_x.data() + offset);
            offset += m_nx;
            d_ocp_qp_sol_get_u(i, &m_qp_sol, this->m_x.data() + offset);
            offset += m_nu;
        }
        d_ocp_qp_sol_get_x(m_N, &m_qp_sol, this->m_x.data() + offset);

        // extract dynamics eq dual variables
        offset = 0;
        for (int i = 0; i < m_N; i++) {
            d_ocp_qp_sol_get_pi(i, &m_qp_sol, this->m_lam.data() + offset);
            offset += m_nx;
        }

        // extract box dual variables
        offset = 0;
        for (int i  = 0; i <= m_N; i++) {
            d_ocp_qp_sol_get_lam_lbx(i, &m_qp_sol, m_lbx.data());
            d_ocp_qp_sol_get_lam_ubx(i, &m_qp_sol, m_ubx.data());
            int lbx_offset = 0;
            int ubx_offset = 0;
            for (int j = 0; j < m_nx; j++) {
                if (this->m_box_constraint_type[offset + j] == constraint_t::EQ_CONSTR ||
                    this->m_box_constraint_type[offset + j] == constraint_t::INEQ_CONSTR)
                {
                    this->m_lam_bounds(offset + j) = m_ubx(ubx_offset++) - m_lbx(lbx_offset++);
                }
                else if (this->m_box_constraint_type[offset + j] == constraint_t::INEQ_LB_ONLY_CONSTR)
                {
                    this->m_lam_bounds(offset + j) = -m_lbx(lbx_offset++);
                }
                else if (this->m_box_constraint_type[offset + j] == constraint_t::INEQ_UB_ONLY_CONSTR)
                {
                    this->m_lam_bounds(offset + j) = m_ubx(ubx_offset++);
                }
                else
                {
                    this->m_lam_bounds(offset + j) = 0;
                }
            }
            offset += m_nx;

            if (i < m_N) {
                d_ocp_qp_sol_get_lam_lbu(i, &m_qp_sol, m_lbu.data());
                d_ocp_qp_sol_get_lam_ubu(i, &m_qp_sol, m_ubu.data());
                int lbu_offset = 0;
                int ubu_offset = 0;
                for (int j = 0; j < m_nu; j++) {
                    if (this->m_box_constraint_type[offset + j] == constraint_t::EQ_CONSTR ||
                        this->m_box_constraint_type[offset + j] == constraint_t::INEQ_CONSTR)
                    {
                        this->m_lam_bounds(offset + j) = m_ubu(ubu_offset++) - m_lbu(lbu_offset++);
                    }
                    else if (this->m_box_constraint_type[offset + j] == constraint_t::INEQ_LB_ONLY_CONSTR)
                    {
                        this->m_lam_bounds(offset + j) = -m_lbu(lbu_offset++);
                    }
                    else if (this->m_box_constraint_type[offset + j] == constraint_t::INEQ_UB_ONLY_CONSTR)
                    {
                        this->m_lam_bounds(offset + j) = m_ubu(ubu_offset++);
                    }
                    else
                    {
                        this->m_lam_bounds(offset + j) = 0;
                    }
                }
                offset += m_nu;
            }
        }

        // extract generic ineq dual variables
        // TODO

        // update status
        switch (hpipm_status)
        {
            case 0:
                this->m_info.status = qp_status_t::SOLVED;
                break;
            case 1:
                this->m_info.status = qp_status_t::MAX_ITER_REACHED;
                break;
            default:
                this->m_info.status = qp_status_t::UNSOLVED;
                break;
        }

        return this->m_info;
    }

private:

};

template<typename T>
struct is_fixed_end_time_multiple_shooting : std::false_type {};

template<typename ControlProblem, unsigned N_segs, int DiffOptions>
struct is_fixed_end_time_multiple_shooting<laopt_tools::MultipleShooting<ControlProblem, N_segs, DiffOptions>> : std::true_type
{
    static_assert((ControlProblem::Options & laopt_tools::FreeEndTime) == 0, "Multiple shooting transcription must be fixed end time");
};

template<typename Problem>
class SQPSolver<Problem, HPIPMSolver> : public SQPBase<SQPSolver<Problem, HPIPMSolver>, Problem, HPIPMSolver>
{
    using Base = SQPBase<SQPSolver<Problem, HPIPMSolver>, Problem, HPIPMSolver>;
    using UserCode = typename Problem::UserCode;
    static_assert(is_fixed_end_time_multiple_shooting<UserCode>::value, "HPIPM only works with the laopt_tools::MultipleShooting");

    using ControlProblem = typename UserCode::ControlProblem;
    static_assert(ControlProblem::NP == 0, "HPIPM doesn't support parameters");

public:
    explicit SQPSolver(Problem& prob) : Base::SQPBase(prob)
    {
        this->m_qp_solver.set_problem_dims(UserCode::N, ControlProblem::NX, ControlProblem::NU,
                                           ControlProblem::NG, ControlProblem::NG0, ControlProblem::NGF);
    }
};

} // namespace laopt

#endif // LAOPT_HPIPM_INTERFACE_HPP
