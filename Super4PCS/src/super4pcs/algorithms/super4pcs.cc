// Copyright 2014 Nicolas Mellado
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
// Authors: Nicolas Mellado, Dror Aiger
//
// An implementation of the Super 4-points Congruent Sets (Super 4PCS)
// algorithm presented in:
//
// Super 4PCS: Fast Global Pointcloud Registration via Smart Indexing
// Nicolas Mellado, Dror Aiger, Niloy J. Mitra
// Symposium on Geometry Processing 2014.
//
// Data acquisition in large-scale scenes regularly involves accumulating
// information across multiple scans. A common approach is to locally align scan
// pairs using Iterative Closest Point (ICP) algorithm (or its variants), but
// requires static scenes and small motion between scan pairs. This prevents
// accumulating data across multiple scan sessions and/or different acquisition
// modalities (e.g., stereo, depth scans). Alternatively, one can use a global
// registration algorithm allowing scans to be in arbitrary initial poses. The
// state-of-the-art global registration algorithm, 4PCS, however has a quadratic
// time complexity in the number of data points. This vastly limits its
// applicability to acquisition of large environments. We present Super 4PCS for
// global pointcloud registration that is optimal, i.e., runs in linear time (in
// the number of data points) and is also output sensitive in the complexity of
// the alignment problem based on the (unknown) overlap across scan pairs.
// Technically, we map the algorithm as an 'instance problem' and solve it
// efficiently using a smart indexing data organization. The algorithm is
// simple, memory-efficient, and fast. We demonstrate that Super 4PCS results in
// significant speedup over alternative approaches and allows unstructured
// efficient acquisition of scenes at scales previously not possible. Complete
// source code and datasets are available for research use at
// http://geometry.cs.ucl.ac.uk/projects/2014/super4PCS/.

#include "super4pcs/algorithms/super4pcs.h"
#include "super4pcs/accelerators/bbox.h"

#ifdef SUPER4PCS_USE_CHEALPIX
#include "super4pcs/accelerators/normalHealSet.h"
#else
#include "super4pcs/accelerators/normalset.h"
#endif

#include <fstream>
#include <array>
#include <time.h>
#include <map>

//#define MULTISCALE

namespace GlobalRegistration {


MatchSuper4PCS::MatchSuper4PCS(const Match4PCSOptions& options,
                               const Utils::Logger &logger)
    : Base(options
           , logger
#ifdef SUPER4PCS_USE_OPENMP
           , 1
#endif
           ),
      pcfunctor_(options_, sampled_Q_3D_) { }

MatchSuper4PCS::~MatchSuper4PCS() { }

/// \brief create connectivity table
/// \param pairs
/// \param table
/// \return
///
void
CreateConnectivityTable(
	const std::vector<std::pair<int, int>>& pairs,
	std::map<int,std::vector<int>>& table)
{
	std::vector<std::pair<int, int>>::const_iterator pair = pairs.begin();
	while (pair != pairs.end())
	{
		//if point1 in pair is in the table keys
		//add point2 to the vector of the table
		if (table.find((*pair).first) != table.end())
			table[(*pair).first].emplace_back((*pair).second);
		//if not
		//create a new key point1 with a vector of only point2
		else
			table.insert(std::pair<int, std::vector<int>>
			((*pair).first, std::vector<int>(1, (*pair).second)));

		//pair doen't have direction, so both point in pair should become a key in table
		if (table.find((*pair).second) != table.end())
			table[(*pair).second].emplace_back((*pair).first);
		else
			table.insert(std::pair<int, std::vector<int>>
			((*pair).second, std::vector<int>(1, (*pair).first)));
		pair++;
	}

	//Sort all the vectors for getting intersection 
	std::map<int, std::vector<int>>::iterator it = table.begin();

	for (; it != table.end(); it++)
	{
	    std::sort((*it).second.begin(), (*it).second.end());
	}
}

bool
Intersect(std::vector<int>& V1, std::vector<int>& V2, std::vector<int>& V12)
{
	std::set_intersection(V1.begin(), V1.end(), V2.begin(), V2.end(), std::inserter(V12, V12.begin()));
	return !V12.empty();
}

//Find all congruent tetrahedron
bool
MatchSuper4PCS::FindCongruentQuadrilaterals(
	Scalar distance_threshold,
	const std::vector<std::pair<int, int>>& pairs1,
	const std::vector<std::pair<int, int>>& pairs2,
	const std::vector<std::pair<int, int>>& pairs3,
	const std::vector<std::pair<int, int>>& pairs4,
	const std::vector<std::pair<int, int>>& pairs5,
	const std::vector<std::pair<int, int>>& pairs6,
	std::vector<Quadrilateral>* quadrilaterals) const {


	typedef PairCreationFunctor<Scalar>::Point Point;
	typedef std::map<int, std::vector<int>> CTable;
#ifdef SUPER4PCS_USE_CHEALPIX
	typedef GlobalRegistration::IndexedNormalHealSet IndexedNormalSet3D;
#else
	typedef  GlobalRegistration::IndexedNormalSet
		< Point,   //! \brief Point type used internally
		3,       //! \brief Nb dimension
		7,       //! \brief Nb cells/dim normal
		Scalar>  //! \brief Scalar type
		IndexedNormalSet3D;
#endif


	if (quadrilaterals == nullptr) return false;

	quadrilaterals->clear();

	// 1. Datastructure construction
	const Scalar eps = pcfunctor_.getNormalizedEpsilon(distance_threshold);

	IndexedNormalSet3D nset(eps);

	//Create connectivity tables 
	CTable table1, table2, table3, table4, table5, table6;
	CreateConnectivityTable(pairs1, table1);
	CreateConnectivityTable(pairs2, table2);
	CreateConnectivityTable(pairs3, table3);
	CreateConnectivityTable(pairs4, table4);
	CreateConnectivityTable(pairs5, table5);
	CreateConnectivityTable(pairs6, table6);


	/******
	*          point4
	*             ..
	*         .  .  .
	*  point1 * .    .
	*       .  .*     .
	*      .  .    *   .
	*     .  .       *  .
	*  point2 ........point3
	*/
	///////////////////////
	//distance sort by:  d1 p12, d2 p13, d3 p14, d4 p23, d5 p24, d6 p34
	// Computes distance between pairs.
	//Search tables for tetrahedral
	CTable::iterator it1;
	std::vector<int>::iterator it2, it3, it4;
	//for every point in table1's key, see it as potential point1
	for (it1 = table1.begin(); it1 != table1.end(); it1++)
	{
		int point1 = (*it1).first;
		//if point1 doesn't belong to any pair with distance13
		//it cannot be a good point1
		if (table2.find(point1) == table2.end())
			continue;
		//if point1 doesn't belong to any pair with distance14
		//it cannot be a good point1
		if (table3.find(point1) == table3.end())
			continue;
		//for every point in pointset which have distance12 with point1
		for (it2 = table1[point1].begin(); it2 != table1[point1].end(); it2++)
		{
			//if point2 doesn't belong to any pair with distance23
			//it cannot be a good point2
			if (table4.find(*it2) == table4.end())
				continue;
			//if point2 doesn't belong to any pair with distance24
			//it cannot be a good point2
			if (table5.find(*it2) == table5.end())
				continue;
			std::vector<int> P3;
			//P3 contains the points
			//which have distance13 with point1
			//and distance23 with point2
			//if P3 is empty, point1&2 are not good
			if (!Intersect(table2[point1], table4[*it2], P3))
				continue;
			//for every point in P3
			for (it3 = P3.begin(); it3 != P3.end(); it3++)
			{
				//if point3 doesn't belong to any pair with distance34
				//it cannot be a good point3
				if (table6.find(*it3) == table6.end())
					continue;
				std::vector<int> P12_4, P4;
				//P12_4 contains points
				//which have distance14 with point1
				//and have diatance24 with point2
				if (!Intersect(table3[point1], table5[*it2], P12_4))
					continue;
				//P4 contains points
				//which is in P12_4
				//and have distance34 with point3
				if (!Intersect(P12_4, table6[*it3], P4))
					continue;
				//if P4 is not empty, then point1~4 can form a congruent tetrahedral
				for (it4 = P4.begin(); it4 != P4.end(); it4++)
				{
					quadrilaterals->emplace_back(point1,*it2,*it3,*it4);
				}
			}
		}
	}
	return quadrilaterals->size() != 0;
}
/*
// Finds congruent candidates in the set Q, given the invariants and threshold
// distances.
bool
MatchSuper4PCS::FindCongruentQuadrilaterals(
        Scalar invariant1,
        Scalar invariant2,
        Scalar distance_threshold1,
        Scalar distance_threshold2,
        const std::vector<std::pair<int, int>>& P_pairs,
        const std::vector<std::pair<int, int>>& Q_pairs,
        std::vector<Quadrilateral>* quadrilaterals) const {

    typedef PairCreationFunctor<Scalar>::Point Point;

#ifdef SUPER4PCS_USE_CHEALPIX
    typedef GlobalRegistration::IndexedNormalHealSet IndexedNormalSet3D;
#else
    typedef  GlobalRegistration::IndexedNormalSet
                    < Point,   //! \brief Point type used internally
                      3,       //! \brief Nb dimension
                      7,       //! \brief Nb cells/dim normal
                      Scalar>  //! \brief Scalar type
    IndexedNormalSet3D;
#endif


  if (quadrilaterals == nullptr) return false;

  quadrilaterals->clear();

  // Compute the angle formed by the two vectors of the basis
  const Scalar alpha =
          (base_3D_[1].pos() - base_3D_[0].pos()).normalized().dot(
          (base_3D_[3].pos() - base_3D_[2].pos()).normalized());

  // 1. Datastructure construction
  const Scalar eps = pcfunctor_.getNormalizedEpsilon(distance_threshold2);

  IndexedNormalSet3D nset (eps);

  for (size_t i = 0; i <  P_pairs.size(); ++i) {
    const Point& p1 = pcfunctor_.points[P_pairs[i].first];
    const Point& p2 = pcfunctor_.points[P_pairs[i].second];
    const Point  n  = (p2 - p1).normalized();

    nset.addElement((p1+ Point::Scalar(invariant1) * (p2 - p1)).eval(), n, i);
  }


  std::set< std::pair<unsigned int, unsigned int > > comb;

  unsigned int j = 0;
  std::vector<unsigned int> nei;
  // 2. Query time
  for (unsigned int i = 0; i < Q_pairs.size(); ++i) {
    const Point& p1 = pcfunctor_.points[Q_pairs[i].first];
    const Point& p2 = pcfunctor_.points[Q_pairs[i].second];

    const VectorType& pq1 = sampled_Q_3D_[Q_pairs[i].first].pos();
    const VectorType& pq2 = sampled_Q_3D_[Q_pairs[i].second].pos();

    nei.clear();

    const Point      query  =  p1 + invariant2 * ( p2 - p1 );
    const VectorType queryQ = pq1 + invariant2 * (pq2 - pq1);

    const Point queryn = (p2 - p1).normalized();

    nset.getNeighbors( query, queryn, alpha, nei);


    VectorType invPoint;
    //const Scalar distance_threshold2s = distance_threshold2 * distance_threshold2;
    for (unsigned int k = 0; k != nei.size(); k++){
      const int id = nei[k];

      const VectorType& pp1 = sampled_Q_3D_[P_pairs[id].first].pos();
      const VectorType& pp2 = sampled_Q_3D_[P_pairs[id].second].pos();

      invPoint = pp1 + (pp2 - pp1) * invariant1;

       // use also distance_threshold2 for inv 1 and 2 in 4PCS
      if ((queryQ-invPoint).squaredNorm() <= distance_threshold2){
          comb.emplace(id, i);
      }
    }
  }

  for (std::set< std::pair<unsigned int, unsigned int > >::const_iterator it =
             comb.cbegin();
       it != comb.cend(); it++){
    const unsigned int & id = (*it).first;
    const unsigned int & i  = (*it).second;

    quadrilaterals->emplace_back(P_pairs[id].first, P_pairs[id].second,
                                 Q_pairs[i].first,  Q_pairs[i].second);
  }

  return quadrilaterals->size() != 0;
}
*/

// Constructs two sets of pairs in Q, each corresponds to one pair in the base
// in P, by having the same distance (up to some tolerantz) and optionally the
// same angle between normals and same color.
void
MatchSuper4PCS::ExtractPairs(Scalar pair_distance,
                             Scalar pair_normals_angle,
                             Scalar pair_distance_epsilon,
                             int base_point1,
                             int base_point2,
                             PairsVector* pairs) const {

  using namespace GlobalRegistration::Accelerators::PairExtraction;

  pcfunctor_.pairs = pairs;

  pairs->clear();
  pairs->reserve(2 * pcfunctor_.points.size());

  pcfunctor_.pair_distance         = pair_distance;
  pcfunctor_.pair_distance_epsilon = pair_distance_epsilon;
  pcfunctor_.pair_normals_angle    = pair_normals_angle;
  pcfunctor_.norm_threshold =
      0.5 * options_.max_normal_difference * M_PI / 180.0;

  pcfunctor_.setRadius(pair_distance);
  pcfunctor_.setBase(base_point1, base_point2, base_3D_);


#ifdef MULTISCALE
  BruteForceFunctor
  <PairCreationFunctor<Scalar>::Primitive, PairCreationFunctor<Scalar>::Point, 3, Scalar> interFunctor;
#else
  IntersectionFunctor
          <PairCreationFunctor<Scalar>::Primitive,
          PairCreationFunctor<Scalar>::Point, 3, Scalar> interFunctor;
#endif

  Scalar eps = pcfunctor_.getNormalizedEpsilon(pair_distance_epsilon);

  interFunctor.process(pcfunctor_.primitives,
                       pcfunctor_.points,
                       eps,
                       50,
                       pcfunctor_);
}




// Initialize all internal data structures and data members.
void
MatchSuper4PCS::Initialize(const std::vector<Point3D>& /*P*/,
                           const std::vector<Point3D>& /*Q*/) {
  pcfunctor_.synch3DContent();
}


} // namespace Super4PCS
