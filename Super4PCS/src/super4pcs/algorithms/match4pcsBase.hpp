// Copyright 2017 Nicolas Mellado
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// -------------------------------------------------------------------------- //
//
// Authors: Dror Aiger, Yoni Weill, Nicolas Mellado
//
// This file is part of the implementation of the 4-points Congruent Sets (4PCS)
// algorithm presented in:
//
// 4-points Congruent Sets for Robust Surface Registration
// Dror Aiger, Niloy J. Mitra, Daniel Cohen-Or
// ACM SIGGRAPH 2008 and ACM Transaction of Graphics.
//
// Given two sets of points in 3-space, P and Q, the algorithm applies RANSAC
// in roughly O(n^2) time instead of O(n^3) for standard RANSAC, using an
// efficient method based on invariants, to find the set of all 4-points in Q
// that can be matched by rigid transformation to a given set of 4-points in P
// called a base. This avoids the need to examine all sets of 3-points in Q
// against any base of 3-points in P as in standard RANSAC.
// The algorithm can use colors and normals to speed-up the matching
// and to improve the quality. It can be easily extended to affine/similarity
// transformation but then the speed-up is smaller because of the large number
// of congruent sets. The algorithm can also limit the range of transformations
// when the application knows something on the initial pose but this is not
// necessary in general (though can speed the runtime significantly).

// Home page of the 4PCS project (containing the paper, presentations and a
// demo): http://graphics.stanford.edu/~niloy/research/fpcs/fpcs_sig_08.html
// Use google search on "4-points congruent sets" to see many related papers
// and applications.

#ifndef _SUPER4PCS_ALGO_MATCH_4PCS_BASE_IMPL_
#define _SUPER4PCS_ALGO_MATCH_4PCS_BASE_IMPL_

#ifndef _SUPER4PCS_ALGO_MATCH_4PCS_BASE_
#include "super4pcs/algorithms/match4pcsBase.h"
#endif

#include <chrono>
#include <atomic>

#include <Eigen/Geometry>                 // MatrixBase.homogeneous()
#include <Eigen/SVD>                      // Transform.computeRotationScaling()

namespace GlobalRegistration{

// The main 4PCS function. Computes the best rigid transformation and transfoms
// Q toward P by this transformation.
template <typename Sampler, typename Visitor>
Match4PCSBase::Scalar
Match4PCSBase::ComputeTransformation(const std::vector<Point3D>& P,
                                     std::vector<Point3D>* Q,
                                     Eigen::Ref<MatrixType> transformation,
                                     const Sampler& sampler,
                                     const Visitor& v) {

  if (Q == nullptr) return kLargeNumber;
  if (P.empty() || Q->empty()) return kLargeNumber;

  init(P, *Q, sampler);

  if (best_LCP_ != Scalar(1.))
    Perform_N_steps(number_of_trials_, transformation, Q, v);

#ifdef TEST_GLOBAL_TIMINGS
  Log<LogLevel::Verbose>( "----------- Timings (msec) -------------" );
  Log<LogLevel::Verbose>( " Total computation time  : ", totalTime   );
  Log<LogLevel::Verbose>( " Total verify time       : ", verifyTime  );
  Log<LogLevel::Verbose>( "    Kdtree query         : ", kdTreeTime  );
  Log<LogLevel::Verbose>( "----------------------------------------" );
#endif

  return best_LCP_;
}



template <typename Sampler>
void Match4PCSBase::init(const std::vector<Point3D>& P,
                         const std::vector<Point3D>& Q,
                         const Sampler& sampler){

#ifdef TEST_GLOBAL_TIMINGS
    kdTreeTime = 0;
    totalTime  = 0;
    verifyTime = 0;
#endif

    const Scalar kSmallError = 0.00001;
    const int kMinNumberOfTrials = 4;
    const Scalar kDiameterFraction = 0.3;

    centroid_P_ = VectorType::Zero();
    centroid_Q_ = VectorType::Zero();

    sampled_P_3D_.clear();
    sampled_Q_3D_.clear();
	verify_P.clear();
	verify_Q.clear();

	if (options_.sample_size2 == 0)
	{
		//P's diameter
		P_diameter_ = 0.0;
		Q_diameter_ = 0.0;
		for (int i = 0; i < kNumberOfDiameterTrials; ++i) {
			int at = randomGenerator_() % P.size();
			int bt = randomGenerator_() % P.size();

			Scalar l = (P[bt].pos() - P[at].pos()).norm();
			if (l > P_diameter_) {
				P_diameter_ = l;
			}
		}
		//Q's diameter
		for (int i = 0; i < kNumberOfDiameterTrials; ++i) {
			int at = randomGenerator_() % Q.size();
			int bt = randomGenerator_() % Q.size();

			Scalar l = (Q[bt].pos() - Q[at].pos()).norm();
			if (l > Q_diameter_) {
				Q_diameter_ = l;
			}
		}

		options_.sample_size2 = (size_t)(options_.sample_size1 * P_diameter_ / Q_diameter_);
	}

    // prepare P : sampled_P for base choosing
    if (P.size() > options_.sample_size1){
        sampler(P, options_, sampled_P_3D_);
    }
    else
    {
        Log<LogLevel::ErrorReport>( "(P) More samples requested than available: use whole cloud" );
        sampled_P_3D_ = P;
    }
	// prepare P: verify_P for verifying
	verify_P = sampled_P_3D_;
	

    // prepare Q
    if (Q.size() > options_.sample_size2){
        std::vector<Point3D> uniform_Q;
        sampler(Q, options_, uniform_Q);
		// prepare Q: verify_Q for verifying
		verify_Q = uniform_Q;

        std::shuffle(uniform_Q.begin(), uniform_Q.end(), randomGenerator_);
        size_t nbSamples = std::min(uniform_Q.size(), options_.sample_size2);
        auto endit = uniform_Q.begin(); std::advance(endit, nbSamples );
        std::copy(uniform_Q.begin(), endit, std::back_inserter(sampled_Q_3D_));
    }
    else
    {
        Log<LogLevel::ErrorReport>( "(Q) More samples requested than available: use whole cloud" );
        sampled_Q_3D_ = Q;
		verify_Q = Q;
    }

	std::cout << "set sample points for P:		"<< options_.sample_size1 << std::endl;
	std::cout << "set sample points for Q:		"<< options_.sample_size2 << std::endl;
	std::cout << "sample points for P during PG:	"<< sampled_P_3D_.size() << std::endl;
	std::cout << "sample points for Q during PG:	"<< sampled_Q_3D_.size() << std::endl;
	std::cout << "sample points for P during CSV:	"<< verify_P.size() << std::endl;
	std::cout << "sample points for Q during CSV:	"<< verify_Q.size() << std::endl;

    // center points around centroids
    auto centerPoints = [](std::vector<Point3D>&container,
            VectorType& centroid){
        for(const auto& p : container) centroid += p.pos();
        centroid /= Scalar(container.size());
        for(auto& p : container) p.pos() -= centroid;
    };
    centerPoints(sampled_P_3D_, centroid_P_);
    centerPoints(sampled_Q_3D_, centroid_Q_);

    initKdTree();

    // Compute the diameter of P approximately (randomly). This is far from being
    // Guaranteed close to the diameter but gives good results for most common
    // objects if they are densely sampled.
    P_diameter_ = 0.0;
    for (int i = 0; i < kNumberOfDiameterTrials; ++i) {
        int at = randomGenerator_() % sampled_P_3D_.size();
        int bt = randomGenerator_() % sampled_P_3D_.size();

        Scalar l = (sampled_P_3D_[bt].pos() - sampled_P_3D_[at].pos()).norm();
        if (l > P_diameter_) {
            P_diameter_ = l;
        }
    }

    // Mean distance and a bit more... We increase the estimation to allow for
    // noise, wrong estimation and non-uniform sampling.
    P_mean_distance_ = MeanDistance();

    // Normalize the delta (See the paper) and the maximum base distance.
    // delta = P_mean_distance_ * delta;
    max_base_diameter_ = P_diameter_;  // * estimated_overlap_;

    // RANSAC probability and number of needed trials.
    Scalar first_estimation =
            std::log(kSmallError) / std::log(1.0 - pow(options_.getOverlapEstimation(),
                                             static_cast<Scalar>(kMinNumberOfTrials)));
    // We use a simple heuristic to elevate the probability to a reasonable value
    // given that we don't simply sample from P, but instead, we bound the
    // distance between the points in the base as a fraction of the diameter.
    number_of_trials_ =
            static_cast<int>(first_estimation * (P_diameter_ / kDiameterFraction) /
                             max_base_diameter_);
    if (number_of_trials_ < kMinNumberOfTrials)
        number_of_trials_ = kMinNumberOfTrials;

	std::cout << "RANSAC iterations:" << (unsigned long)number_of_trials_ << std::endl;

    Log<LogLevel::Verbose>( "norm_max_dist: ", options_.delta );
    current_trial_ = 0;
    best_LCP_ = 0.0;

    Q_copy_ = Q;
    for (int i = 0; i < 4; ++i) {
        base_[i] = 0;
        current_congruent_[i] = 0;
    }
    transform_ = Eigen::Matrix<Scalar, 4, 4>::Identity();

    // call Virtual handler
    Initialize(P, Q);

    best_LCP_ = Verify(transform_);
    Log<LogLevel::Verbose>( "Initial LCP: ", best_LCP_ );
}


// Performs N RANSAC iterations and compute the best transformation. Also,
// transforms the set Q by this optimal transformation.
template <typename Visitor>
bool
Match4PCSBase::Perform_N_steps(int n,
                               Eigen::Ref<MatrixType> transformation,
                               std::vector<Point3D>* Q,
                               const Visitor &v) {
  using std::chrono::system_clock;
  if (Q == nullptr) return false;

#ifdef TEST_GLOBAL_TIMINGS
    Timer t (true);
#endif


  // The transformation has been computed between the two point clouds centered
  // at the origin, we need to recompute the translation to apply it to the original clouds
  auto getGlobalTransform = [this](Eigen::Ref<MatrixType> transformation){
    Eigen::Matrix<Scalar, 3, 3> rot, scale;
    Eigen::Transform<Scalar, 3, Eigen::Affine> (transform_).computeRotationScaling(&rot, &scale);
    transformation = transform_;
    transformation.col(3) = (qcentroid1_ + centroid_P_ - ( rot * scale * (qcentroid2_ + centroid_Q_))).homogeneous();
  };

  Scalar last_best_LCP = best_LCP_;
  v(0, best_LCP_, transformation);

  bool ok = false;
  std::chrono::time_point<system_clock> t0 = system_clock::now(), end;


  for (int i = current_trial_; i < current_trial_ + n; ++i) {
    
	clock_t startTryOneBase, endTryOneBase;
	startTryOneBase = clock();

	ok = TryOneBase(v);

    Scalar fraction_try  = Scalar(i) / Scalar(number_of_trials_);
    Scalar fraction_time =
        std::chrono::duration_cast<std::chrono::seconds>
        (system_clock::now() - t0).count() /
                          options_.max_time_seconds;
    Scalar fraction = std::max(fraction_time, fraction_try);

    if (v.needsGlobalTransformation()) {
      getGlobalTransform(transformation);
    } else {
      transformation = transform_;
    }

    v(fraction, best_LCP_, transformation);

	endTryOneBase = clock();
	double timeOfTryOneBase = (double)(endTryOneBase - startTryOneBase);
	if (time_display)
	std::cout << "single RANSAC iteration time:" << timeOfTryOneBase << "ms" << std::endl;

    // ok means that we already have the desired LCP.
    if (ok || i > number_of_trials_ || fraction >= 0.99 || best_LCP_ == 1.0) break;
  }

  current_trial_ += n;
  std::cout << "RANSAC iterations:" << current_trial_ << std::endl;
  if (best_LCP_ > last_best_LCP) {
    *Q = Q_copy_;

    getGlobalTransform(transformation);

    // Transforms Q by the new transformation.
    for (size_t i = 0; i < Q->size(); ++i) {
      (*Q)[i].pos() = (transformation * (*Q)[i].pos().homogeneous()).head<3>();
    }
  }
#ifdef TEST_GLOBAL_TIMINGS
    totalTime += Scalar(t.elapsed().count()) / Scalar(CLOCKS_PER_SEC);
#endif

  return ok || current_trial_ >= number_of_trials_;
}



// Pick one base, finds congruent 4-points in Q, verifies for all
// transformations, and retains the best transformation and LCP. This is
// a complete RANSAC iteration.
template<typename Visitor>
bool Match4PCSBase::TryOneBase(const Visitor &v) {
  Scalar invariant1, invariant2;
  int base_id1, base_id2, base_id3, base_id4;

//#define STATIC_BASE

#ifdef STATIC_BASE
  static bool first_time = true;

  if (first_time){
      base_id1 = 0;
      base_id2 = 3;
      base_id3 = 1;
      base_id4 = 4;

      base_3D_[0] = sampled_P_3D_ [base_id1];
      base_3D_[1] = sampled_P_3D_ [base_id2];
      base_3D_[2] = sampled_P_3D_ [base_id3];
      base_3D_[3] = sampled_P_3D_ [base_id4];

      TryQuadrilateral(&invariant1, &invariant2, base_id1, base_id2, base_id3, base_id4);

      first_time = false;
  }
  else
      return false;

#else

  if (!SelectQuadrilateral(base_id1, base_id2,
                           base_id3, base_id4)) {
    return false;
  }
#endif

  /******
  *          point3
  *             ..
  *         .  .  .
  *  point0 * .    .
  *       .  .*     .
  *      .  .    *   .
  *     .  .       *  .
  *  point1 ........point2
  */
  ///////////////////////
  //distance sort by:  p01, p02, p03, p12, p13, p23
  // Computes distance between pairs.
  const Scalar distance1 = (base_3D_[0].pos() - base_3D_[1].pos()).norm();
  const Scalar distance2 = (base_3D_[0].pos() - base_3D_[2].pos()).norm();
  const Scalar distance3 = (base_3D_[0].pos() - base_3D_[3].pos()).norm();
  const Scalar distance4 = (base_3D_[1].pos() - base_3D_[2].pos()).norm();
  const Scalar distance5 = (base_3D_[1].pos() - base_3D_[3].pos()).norm();
  const Scalar distance6 = (base_3D_[2].pos() - base_3D_[3].pos()).norm();

  std::vector<std::pair<int, int>> pairs1, pairs2, pairs3, pairs4, pairs5, pairs6;
  std::vector<Quadrilateral> congruent_quads;

  // Compute normal angles.
  const Scalar normal_angle1 = (base_3D_[0].normal() - base_3D_[1].normal()).norm();
  const Scalar normal_angle2 = (base_3D_[0].normal() - base_3D_[2].normal()).norm();
  const Scalar normal_angle3 = (base_3D_[0].normal() - base_3D_[3].normal()).norm();
  const Scalar normal_angle4 = (base_3D_[1].normal() - base_3D_[2].normal()).norm();
  const Scalar normal_angle5 = (base_3D_[1].normal() - base_3D_[3].normal()).norm();
  const Scalar normal_angle6 = (base_3D_[2].normal() - base_3D_[3].normal()).norm();

  clock_t startExPairs, endExPairs;
  startExPairs = clock();

  ExtractPairs(distance1, normal_angle1, distance_factor * options_.delta, 0,
                  1, &pairs1);
  ExtractPairs(distance2, normal_angle2, distance_factor * options_.delta, 2,
                  3, &pairs2);
  ExtractPairs(distance3, normal_angle3, distance_factor * options_.delta, 0,
	  1, &pairs3);
  ExtractPairs(distance4, normal_angle4, distance_factor * options_.delta, 0,
	  1, &pairs4);
  ExtractPairs(distance5, normal_angle5, distance_factor * options_.delta, 0,
	  1, &pairs5);
  ExtractPairs(distance6, normal_angle6, distance_factor * options_.delta, 0,
	  1, &pairs6);

  endExPairs = clock();
  double timeOfExPairs = (double)(endExPairs - startExPairs);
  if (time_display)
  std::cout << "  time of pair generation: " << timeOfExPairs << "ms" << std::endl;

  std::cout << "Pair creation ouput: \n";
  std::cout << "	Distance	Size" << std::endl;
  std::cout << "Pair1	" << distance1 << "	" << pairs1.size() << std::endl;
  std::cout << "Pair2	" << distance2 << "	" << pairs2.size() << std::endl;
  std::cout << "Pair3	" << distance3 << "	" << pairs3.size() << std::endl;
  std::cout << "Pair4	" << distance4 << "	" << pairs4.size() << std::endl;
  std::cout << "Pair5	" << distance5 << "	" << pairs5.size() << std::endl;
  std::cout << "Pair6	" << distance6 << "	" << pairs6.size() << std::endl;

  if (pairs1.size() == 0 || pairs2.size() == 0
	  || pairs3.size() == 0 || pairs4.size() == 0
	  || pairs5.size() == 0 || pairs6.size() == 0) {
    return false;
  }

  clock_t startFindCS, endFindCS;
  startFindCS = clock();

  if (!FindCongruentQuadrilaterals(distance_factor * options_.delta,
                                   pairs1,
                                   pairs2,
								   pairs3,
								   pairs4,
							       pairs5,
								   pairs6,
                                   &congruent_quads)) {
    return false;
  }
  endFindCS = clock();
  double timeOfFindCS = (double)(endFindCS - startFindCS);
  if (time_display) {
	  std::cout << "  time of CSE:" << timeOfFindCS << "ms" << std::endl;
	  std::cout << "  number of CS:" << congruent_quads.size() << std::endl;
  }
  std::cout << "Mark1" << std::endl;
  size_t nb = 0;

  clock_t startTryCS, endTryCS;
  startTryCS = clock();

  bool match = TryCongruentSet(base_id1, base_id2, base_id3, base_id4,
                               congruent_quads,
                               v,
                               nb);
  std::cout << "Mark2" << std::endl;
  endTryCS = clock();
  double timeOfTryCS = (double)(endTryCS - startTryCS);
  if (time_display)
  std::cout << "  time of CSV:" << timeOfTryCS << "ms" << std::endl;

  if (nb != 0)
	  std::cout << "Congruent quads:"<< nb <<"    " <<std::endl;

  return match;
}


template <typename Visitor>
bool Match4PCSBase::TryCongruentSet(
        int base_id1,
        int base_id2,
        int base_id3,
        int base_id4,
        const std::vector<Quadrilateral>& congruent_quads,
        const Visitor& v,
        size_t &nbCongruent){
    static const double pi = std::acos(-1);

    // get references to the basis coordinates
    const Point3D& b1 = sampled_P_3D_[base_id1];
    const Point3D& b2 = sampled_P_3D_[base_id2];
    const Point3D& b3 = sampled_P_3D_[base_id3];
    const Point3D& b4 = sampled_P_3D_[base_id4];

    // set the basis coordinates in the congruent quad array
    const std::array<Point3D, 4> congruent_base {{b1, b2, b3, b4}};


    // Centroid of the basis, computed once and using only the three first points
    Eigen::Matrix<Scalar, 3, 1> centroid1 = (b1.pos() + b2.pos() + b3.pos()) / Scalar(3);


    std::atomic<size_t> nbCongruentAto(0);

#ifdef SUPER4PCS_USE_OPENMP
#pragma omp parallel for num_threads(omp_nthread_congruent_)
#endif
    for (int i = 0; i < int(congruent_quads.size()); ++i) {
      std::array<Point3D, 4> congruent_candidate;

      Eigen::Matrix<Scalar, 4, 4> transform;

      // Centroid of the sets, computed in the loop using only the three first points
      Eigen::Matrix<Scalar, 3, 1> centroid2;

      const int a = congruent_quads[i].vertices[0];
      const int b = congruent_quads[i].vertices[1];
      const int c = congruent_quads[i].vertices[2];
      const int d = congruent_quads[i].vertices[3];
      congruent_candidate[0] = sampled_Q_3D_[a];
      congruent_candidate[1] = sampled_Q_3D_[b];
      congruent_candidate[2] = sampled_Q_3D_[c];
      congruent_candidate[3] = sampled_Q_3D_[d];

  #ifdef STATIC_BASE
      Log<LogLevel::Verbose>( "Ids: ", base_id1, "\t", base_id2, "\t", base_id3, "\t", base_id4);
      Log<LogLevel::Verbose>( "     ", a, "\t", b, "\t", c, "\t", d);
  #endif

      centroid2 = (congruent_candidate[0].pos() +
                   congruent_candidate[1].pos() +
                   congruent_candidate[2].pos()) / Scalar(3.);

      Scalar rms = -1;

      const bool ok =
      ComputeRigidTransformation(congruent_base,     // input congruent quad
                                 congruent_candidate,// tested congruent quad
                                 centroid1,          // input: basis centroid
                                 centroid2,          // input: candidate quad centroid
                                 options_.max_angle * pi / 180.0, // maximum per-dimension angle, check return value to detect invalid cases
                                 transform,          // output: transformation
                                 rms,                // output: rms error of the transformation between the basis and the congruent quad
        //                     #ifdef MULTISCALE
                                 true
         //                    #else
         //                       false
         //                    #endif
                                 );             // state: compute scale ratio ?

      if (ok && rms >= Scalar(0.)) {

        // We give more tolerantz in computing the best rigid transformation.
        if (rms < distance_factor * options_.delta) {

          nbCongruentAto++;
          // The transformation is computed from the point-clouds centered inn [0,0,0]

          // Verify the rest of the points in Q against P.
          Scalar lcp = Verify(transform);

          // transformation has been computed between the two point clouds centered
          // at the origin, we need to recompute the translation to apply it to the original clouds
          auto getGlobalTransform =
              [this, transform, centroid1, centroid2]
              (Eigen::Ref<MatrixType> transformation){
            Eigen::Matrix<Scalar, 3, 3> rot, scale;
            Eigen::Transform<Scalar, 3, Eigen::Affine> (transform).computeRotationScaling(&rot, &scale);
            transformation = transform;
            transformation.col(3) = (centroid1 + centroid_P_ - ( rot * scale * (centroid2 + centroid_Q_))).homogeneous();
          };

          if (v.needsGlobalTransformation())
          {
            Eigen::Matrix<Scalar, 4, 4> transformation = transform;
            getGlobalTransform(transformation);
            v(-1, lcp, transformation);
          }
          else
            v(-1, lcp, transform);

#pragma omp critical
          if (lcp > best_LCP_) {
            // Retain the best LCP and transformation.
            base_[0] = base_id1;
            base_[1] = base_id2;
            base_[2] = base_id3;
            base_[3] = base_id4;

            current_congruent_[0] = a;
            current_congruent_[1] = b;
            current_congruent_[2] = c;
            current_congruent_[3] = d;

            best_LCP_    = lcp;
            transform_   = transform;
            qcentroid1_  = centroid1;
            qcentroid2_  = centroid2;
          }
          // Terminate if we have the desired LCP already.
          if (best_LCP_ > options_.getTerminateThreshold()){
            continue;
          }
        }
      }
    }

    nbCongruent = nbCongruentAto;

    // If we reached here we do not have yet the desired LCP.
    return best_LCP_ > options_.getTerminateThreshold() /*false*/;
}


} // namespace Super4PCS

#endif
