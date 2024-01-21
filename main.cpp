#include <iostream>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>
#include <unordered_set>
#include <utility>
#include <random>
#include <cmath>
#include "mpi.h"
#include <algorithm>
#include <unistd.h>

bool hardCode = false;
bool verbose = false;
bool verbose_1 = false;
bool verbose_2 = false;
bool verbose_3 = false;
bool verbose_4 = true;

#define dumpAll \
  if (verbose_4) {\
bdVetSet.clear();\
inVetSet.clear();\
extVetSet.clear();\
checkInBdExt(inVetSet, bdVetSet, extVetSet, adjTab, vtxdist, mpi_rank);\
for (int rank = 0; rank < mpi_size; rank++) {\
sleep(1);\
MPI_Barrier(MPI_COMM_WORLD);\
if (mpi_rank == rank) {\
cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl;\
cout << endl << "rank: " << mpi_rank << endl;\
cout << "--------------------------------" << endl;\
cout <<  "Cross rank give/get info:" << endl;\
cout << "--------------------------------" << endl;\
cout << endl << "getTab:" << endl;\
for (auto &val: getTab) {\
cout << "Vet: " << val.second << "; get the Vet: " << val.first << endl;\
}\
cout << endl << "giveTab:" << endl;\
\
for (auto &val: giveTab) {\
cout << "Vet: " << val.first << "; is given to Vet: " << val.second << endl;\
}\
\
\
cout << "--------------------------------" << endl;\
cout <<  "adj info:" << endl;\
cout << "--------------------------------" << endl;\
for (auto &val: adjTab) {\
cout << "vet: " << val.first << " is adj to: " << endl;\
for (auto &adjVet: val.second) {\
cout << adjVet << ", ";\
}\
cout << endl;\
}\
cout << "--------------------------------" << endl;\
cout <<  "wgt info:" << endl;\
cout << "--------------------------------" << endl;\
\
for (auto &val: edgeWgtTab) {\
cout << "edge: " << key2a(val.first) << "-" << key2b(val.first) << ": " << val.second << endl;\
}\
cout << "--------------------------------" << endl;\
cout << "set Info" << endl;\
cout << "--------------------------------" << endl;\
cout << "inVetSet:" << endl;\
for (auto& inVet : inVetSet){\
cout << inVet << ", " ;\
}\
cout << endl;\
\
cout << "bdVetSet:" << endl;\
for (auto& bdVet : bdVetSet){\
cout << bdVet << ", " ;\
}\
cout << endl;\
\
cout << "extVetSet:" << endl;\
for (auto& extVet : extVetSet){\
cout << extVet << ", " ;\
}\
cout << endl;\
\
\
\
cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl << endl << endl;\
\
}\
}\
}           

const int MAGIC_NUM= -999;

using namespace std;

struct writeMatch{ // the edge weight should be available for both vets
  int v_to_write; // which vet to be writted?
  int v_writor; // which vet wants to write to v_to_write? the value requested to be written is actually v_writor
}; // if y_to_write = magic_num, y_writor is the number of writes in this pid's stripe

struct readMatch{
  int v_to_read; // want to read who?
  int v_reader; // who want to read?
  int value; // what is the readed value?
  int who_keeps; // if write successfully (value = v_reader), should tell which v/side keeps, otherwise, default = -1
}; //if y_to_read = magic_num, y_reader is the number of writes in this pid's stripe

struct vetWgt{ // subgraph of giving / recv adj
  int v_adj;
  int wgt;
};

struct matchRes {
  int val;
  int isGiven; // if this is 1, means given, to other vertex
};

typedef unordered_map<int, unordered_set<int>> adjTabType;
typedef pair<const int, unordered_set<int>> oneAdjTabType;
typedef vector<oneAdjTabType*> adjTabPtrVecType;
typedef unordered_map<size_t, int> edgeWgtTabType;

// give m * n, return serial csr for n by m structured mesh, with uniform weight
// for element i,
// its adjacency index are from adjncy[xadj[i]] to adjncy[xadj[i+1]-1]
// its corresponding edge weight are from adjwgt[xadj[i]] to adjwgt[xadj[i+1]-1]
void get_serial_csr(int m, int n, vector<int>& xadj, vector<int>& adjncy, vector<int>& adjwgt){
  int cur_idx = 0;
  int cur_count = 0;
  int prev_xadj = 0;
  xadj.push_back(0);
  for(int i = 0; i < n; i++){
    for (int j = 0; j < m; j++){
      int idx_u = cur_idx - m;
      int idx_d = cur_idx + m;
      int idx_l = cur_idx - 1;
      int idx_r = cur_idx + 1;

      bool left = (j == 0);
      bool right = (j == (m-1));
      bool up = (i == 0);
      bool down = (i == (n-1));

      if(left) idx_l = -1;
      if(right) idx_r = -1;
      if(down) idx_d = -1;
      if(up) idx_u = -1;

      vector<int> idx_list = {idx_u, idx_l, idx_r, idx_d};
      for (auto& idx_adj: idx_list){
        if (idx_adj >= 0){
          cur_count++;
          adjncy.push_back(idx_adj);
          adjwgt.push_back(idx_adj + cur_idx); // set edge weight to be the sum of two vet id
        }
      }
      int next_xadj = prev_xadj + cur_count;
      xadj.push_back(next_xadj);

      cur_idx++;
    }
  }
}

// key value
inline size_t key(int a, int b) {return (size_t) a << 32 | (unsigned int) b;}
// get back
inline int key2a(size_t key) {return key >> 32;}
inline int key2b(size_t key) {return key & 0xFFFFFFFF;}


// get adjList
inline void csr_to_adjList(adjTabType& adjTab, edgeWgtTabType& edgeWgtTab,
                    int mpi_rank, int mpi_size,
                    vector<int>& xadj, vector<int>& adjncy, vector<int>&  adjwgt, vector<int>& vtxdist){
  int offset = vtxdist[mpi_rank];
  int in_vet_min = vtxdist[mpi_rank];
  int in_vet_max = vtxdist[mpi_rank + 1];
  for (int idx = 0; idx < xadj.size() - 1; idx++){
    int cur_vet = idx + offset;
    for (int adj_idx = xadj[idx]; adj_idx < xadj[idx+1]; adj_idx++) {
      adjTab[cur_vet].insert(adjncy[adj_idx]);
      if (cur_vet >= in_vet_min && cur_vet < in_vet_max)
        edgeWgtTab[key(cur_vet, adjncy[adj_idx])] = adjwgt[adj_idx]; // this step introduce zero for key(extVet, in/bdVet), which is not ideal....
    }
  }
}

// checkInBdExt
inline void checkInBdExt(unordered_set<int>& inVetSet, unordered_set<int>& bdVetSet, unordered_set<int>& extVetSet,
                         adjTabType& adjTab, vector<int>& vtxdist, int mpi_rank){
  int in_vet_min = vtxdist[mpi_rank];
  int in_vet_max = vtxdist[mpi_rank + 1];
  for (auto& pair : adjTab){
    int cur_vet = pair.first;
    bool allEdgeIsInternal = true;
    for (auto& adj_vet : pair.second){
      if (adj_vet < in_vet_min || adj_vet >= in_vet_max){ // external edge! + boundary nodes
        extVetSet.insert(adj_vet);
        bdVetSet.insert(cur_vet);
        allEdgeIsInternal = false; // edge is external...
      }
    }
    if (allEdgeIsInternal){
      inVetSet.insert(cur_vet);
    }
  }
}

// get where vet from
inline int whichPid(int vet, vector<int>& vtxdist){
  for (int pid = 0; pid < vtxdist.size()-1 ; pid++){
    if (vet >= vtxdist[pid] && vet < vtxdist[pid+1]){
      return pid;
    }
  }
  exit(2);
}

// give a vet, find HE
inline int findHE(int v_cur, adjTabType& adjTab, edgeWgtTabType& edgeWgtTab, unordered_map<int, int>& match){
  set<pair<int, int>> wgt_v_adj_ranking;
  // temporarily forget the random for the tied elemnt
//        vector<int> randIdx;
//        int n_adj = adjTab[v_cur].size();
//        for (int i = 0; i < n_adj; i++){
//          randIdx.push_back(i);
//        }
//        shuffle(randIdx.begin(), randIdx.end(), gen);

  for (auto& v_adj : adjTab[v_cur]){
//          int v_adj = adjTab[v_cur][randIdx[i]];

    if(match[v_adj] == -1) {
      pair<int, int> cur_wgt_adj(edgeWgtTab[key(v_cur, v_adj)], v_adj);
      wgt_v_adj_ranking.insert(cur_wgt_adj);
    }
  }

  int v_matched = (wgt_v_adj_ranking.rbegin())->second;
  if (v_cur == 3) {cout << "vet 3's matched is: " << v_matched << endl << "match[8] val is: " << match[8] << endl;}
  return v_matched;
}

int main(int argc, char** argv) {

  vector<int> xadj_serial;
  vector<int> adjncy_serial;
  vector<int> adjwgt_serial;

  // tested structured mesh
  int m = 5;
  int n = 4;
  get_serial_csr(m, n, xadj_serial, adjncy_serial, adjwgt_serial);

  // build the distributed graph
  MPI_Init(&argc, &argv);

  int mpi_rank;
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  if (verbose) {
    if (mpi_rank == 0) {
      cout << "number of ranks: " << mpi_size << endl;
      cout << endl << "xadj-serial:" << endl;
      for (auto &val: xadj_serial) cout << val << ", ";
      cout << endl << "adjncy-serial:" << endl;
      for (auto &val: adjncy_serial) cout << val << ", ";
      cout << endl << "adjwgt-serial:" << endl;
      for (auto &val: adjwgt_serial) cout << val << ", ";
      cout << endl;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);


  int graph_serial_size = xadj_serial.size() - 1;
  int start_xadj_idx =  int(ceil(double(graph_serial_size) / mpi_size)) * mpi_rank;

//  cout << "rank: " << mpi_rank << ", start_xadj_idx is:" << start_xadj_idx << endl;

  MPI_Barrier(MPI_COMM_WORLD);

  int start_xadj_val = xadj_serial[start_xadj_idx];
  int num_xadj_ele; // actually this is number of element of local vets
//  cout << "size is:" << int(ceil(double(graph_serial_size )/ mpi_size)) << endl;
  if (mpi_rank == mpi_size - 1){
    num_xadj_ele = graph_serial_size - int(ceil(double(graph_serial_size) / mpi_size)) * mpi_rank; // if not fully divided
  } else {
    num_xadj_ele = int(ceil(double(graph_serial_size) / mpi_size));
  }

  int end_xadj_idx = start_xadj_idx + num_xadj_ele;
  int end_xadj_val = xadj_serial[end_xadj_idx];

//  cout << "rank: " << mpi_rank << ", end_xadj_idx is:" << end_xadj_idx << endl;
  MPI_Barrier(MPI_COMM_WORLD);

  vector<int> xadj(xadj_serial.begin() + start_xadj_idx, xadj_serial.begin() + end_xadj_idx + 1);
  vector<int> adjncy(adjncy_serial.begin() + start_xadj_val, adjncy_serial.begin() + end_xadj_val);
  vector<int> adjwgt(adjwgt_serial.begin() + start_xadj_val, adjwgt_serial.begin() + end_xadj_val);
  for (auto& val : xadj) val -= start_xadj_val;
  vector<int> vtxdist;
  vtxdist.push_back(0);
  int cur_i = 1;
  while (vtxdist.back() < graph_serial_size){
    int tail_idx = cur_i * int(ceil(double(graph_serial_size) / mpi_size));
    vtxdist.push_back(tail_idx);
    cur_i ++;
  }

  if (verbose) {
    // sequential printing
    for (int rank = 0; rank < mpi_size; rank++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (mpi_rank == rank) {
        cout << endl << "rank: " << mpi_rank << endl;
        cout << endl << "xadj:" << endl;
        for (auto &val: xadj) cout << val << ", ";
        cout << endl << "adjncy:" << endl;
        for (auto &val: adjncy) cout << val << ", ";
        cout << endl << "adjwgt:" << endl;
        for (auto &val: adjwgt) cout << val << ", ";
        cout << endl << "vtxdist" << endl;
        for (auto &val: vtxdist) cout << val << ",";
        cout << endl;
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  /* preprocess */
  // csr to adjlist
  adjTabType adjTab;
  edgeWgtTabType edgeWgtTab;
  csr_to_adjList(adjTab, edgeWgtTab, mpi_rank, mpi_size, xadj, adjncy, adjwgt, vtxdist);

  // sequential printing
  if (verbose) {
    for (int rank = 0; rank < mpi_size; rank++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (mpi_rank == rank) {
        cout << endl << "rank: " << mpi_rank << endl;
        cout << endl << "adj info:" << endl;
        for (auto &val: adjTab) {
          cout << "vet: " << val.first << " is adj to: " << endl;
          for (auto &adjVet: val.second) {
            cout << adjVet << ", ";
          }
          cout << endl;
        }
        cout << endl << "wgt info:" << endl;

        for (auto &val: edgeWgtTab) {
          cout << "edge: " << key2a(val.first) << "-" << key2b(val.first) << ": " << val.second << endl;
        }
        cout << endl;

      }
    }
  }

  // format of the data send and recv for writing and reading stage
  MPI_Datatype mpi_write_type;
  MPI_Datatype mpi_read_type;
  MPI_Type_contiguous(2, MPI_INT, &mpi_write_type);
  MPI_Type_commit(&mpi_write_type);

  MPI_Type_contiguous(4, MPI_INT, &mpi_read_type);
  MPI_Type_commit(&mpi_read_type);

  MPI_Datatype mpi_vetWgt_type;
  MPI_Type_contiguous(2, MPI_INT, &mpi_vetWgt_type);
  MPI_Type_commit(&mpi_vetWgt_type);

  MPI_Datatype mpi_match_res_type;
  MPI_Type_contiguous(2, MPI_INT, &mpi_match_res_type);
  MPI_Type_commit(&mpi_match_res_type);



  // label internal and boundary nodes -> gives the following data structure
//  adjTabPtrVecType inVetList;
//  adjTabPtrVecType bdVetList;
  unordered_set<int> inVetSet;
  unordered_set<int> bdVetSet;
  unordered_set<int> extVetSet;

  checkInBdExt(inVetSet, bdVetSet, extVetSet, adjTab, vtxdist, mpi_rank);
  MPI_Barrier(MPI_COMM_WORLD);
  if (verbose){
    for (int rank = 0; rank < mpi_size; rank++){
      MPI_Barrier(MPI_COMM_WORLD);
      if (mpi_rank == rank) {
        cout << "rank: " << mpi_rank << endl;
        cout << "internal vet info: " << endl;
        for (auto& val : inVetSet){
          cout << val<< ", ";
        }
        cout << endl;
        cout << "boundary vet info: " << endl;
        for (auto& val : bdVetSet){
          cout << val << ", ";
        }
        cout << endl << endl;
      }
    }
  }

  /* set up communication */
  // because the internal vertices are evenly distributed, therefore, using vtxdist, by defining an ascending order,
  // we will be able to set up communication without the need of communication

  // send & recv should strictly follow the sequence of bdVetPid, default to ordered_map
  // idx = pid, val = extVets, extVet should be unique for whole 2d vector
  vector<set<int>> extVetPid; // extVet from which pid? used to recv ext vet data from other pid
  extVetPid.resize(mpi_size);
  vector<int> extVetDisp(mpi_size);
  vector<int> extVetNum(mpi_size);

  // idx = pid, val = bdVet, bdVet is unique for each row of vector, but not across whole 2d vector
  vector<set<int>> bdVetPid; // bdVet connect ext in which pid? used to send bd data to other pid
  bdVetPid.resize(mpi_size);
  vector<int> bdVetDisp(mpi_size);
  vector<int> bdVetNum(mpi_size);


  for (auto& extVet : extVetSet){
    int pid = whichPid(extVet, vtxdist);
    extVetPid[pid].insert(extVet);
    extVetNum[pid]++;
  }
  int cur_disp = 0;
  for (int i = 0; i < mpi_size; i++){
    extVetDisp[i] = cur_disp;
    cur_disp += extVetNum[i];
  }

  for (auto& bdVet : bdVetSet){
    for (auto& extVet : adjTab[bdVet]){
      int pid = whichPid(extVet, vtxdist);
      if (pid != mpi_rank){
        bdVetPid[pid].insert(bdVet);
//        bdVetNum[pid]++; // oops, leads to bug! non-duplicated!!!!
      }
    }
  }
  for (int pid = 0; pid < mpi_size; pid++){
    bdVetNum[pid] = bdVetPid[pid].size();
  }

  cur_disp = 0;
  for (int i = 0; i < mpi_size; i++){
    bdVetDisp[i] = cur_disp;
    cur_disp += bdVetNum[i];
  }

  // unroll the 2d vectors of ext and bd vetPid, easy to use
  vector<int> extVetPid1d;
  vector<int> bdVetPid1d;
  for (auto& set_pid : extVetPid){
    for (auto& extVet : set_pid){
      extVetPid1d.push_back(extVet);
    }
  }
  for (auto& set_pid : bdVetPid){
    for (auto& bdVet : set_pid){
      bdVetPid1d.push_back(bdVet);
    }
  }


  if(verbose_1){
    for (int rank = 0; rank < mpi_size; rank++){
      MPI_Barrier(MPI_COMM_WORLD);
      if (mpi_rank == rank) {
        cout << "--------------------------" << endl;
        cout << "rank: " << mpi_rank << endl;
        cout << "extWhere: " << endl;
        int i = 0;
        for (auto& val : extVetPid){
          cout << "pid: " << i << ": ";
          i++;
          for (auto& ele : val){
            cout << ele << ", ";
          }
          cout << endl;
        }
        cout << "extnum: " << endl;
        for (auto& val : extVetNum){
          cout << val << ", " ;
        }
        cout << endl;

        cout << "extdisp: " << endl;
        for (auto& val : extVetDisp){
          cout << val << ", " ;
        }
        cout << endl;

        cout << "bdWhere: " << endl;
        i = 0;
        for (auto& val : bdVetPid){
          cout << "pid: " << i << ": ";
          i++;
          for (auto& ele : val){
            cout << ele << ", ";
          }
          cout << endl;
        }
        cout << "bdnum: " << endl;
        for (auto& val : bdVetNum){
          cout << val << ", " ;
        }
        cout << endl;

        cout << "bddisp: " << endl;
        for (auto& val : bdVetDisp){
          cout << val << ", " ;
        }
        cout << endl;
      }
    }
  }


  /* graph coloring */
  double fracLim = 1.0;
  if (fracLim >1 || fracLim <=0){
    MPI_Abort(MPI_COMM_WORLD, 10);
  }


  std::mt19937 gen;
  gen.seed(123 + mpi_rank); // assign seed to each rank
  uniform_int_distribution dis_int_1N4 = uniform_int_distribution(1, graph_serial_size*graph_serial_size*graph_serial_size*graph_serial_size);
  uniform_int_distribution dis_int_01 = uniform_int_distribution(0, 1);

// using macro is more elegant
  vector<int> vetValVec(num_xadj_ele);
  vector<int> vetColorVec(num_xadj_ele);
#define vetVal(i) vetValVec[(i) - vtxdist[mpi_rank]]
#define vetColor(i) vetColorVec[(i) - vtxdist[mpi_rank]]
  vector<vector<int>> colorInVet; // 2d array, idx is the color, color[idx] is each array
  vector<vector<int>> colorBdVet;
  vector<vector<int>> colorExtVet;
  unordered_map<int, int> extVetColor;


  int cur_color = 1;
  bool stop = false;
  int colored_num_total_new = 0;
  int colored_num_total_old = 0;

  int cur_iter = 0;

  while(!stop){
    cur_iter++;

//    if (mpi_rank == 0) {
//      cout << "++++++++++++++++++++++++++++++++++" << endl << "Cur Iter: " << cur_iter << " ; Cur Color: " << cur_color << endl
//           << "=================================+" << endl << endl;
//      cout << "colored_num_total_new: " << colored_num_total_new << "; colored_num_total_old: " << colored_num_total_old << endl;
//    }

    if (colored_num_total_new != colored_num_total_old) cur_color++; // only increase color when new vet have actually been colored
    colored_num_total_old = colored_num_total_new;

    vector<int> inVetThisColor; // store the internal vets for this color, empty if no vet colored
    vector<int> bdVetThisColor; // store the boundary vets for this color, empty if no vet colored

//    if (mpi_rank == 0) {
//      cout << "----------------------------------" << endl << "Cur Iter: " << cur_iter << " ; Cur Color: " << cur_color << endl
//           << "=================================+" << endl << endl;
//      cout << "colored_num_total_new: " << colored_num_total_new << "; colored_num_total_old: " << colored_num_total_old << endl;
//    }

    // assign a random val to bd key-value pair in vetVal
    // because bdVetList and inVetList point to different key values
    for (auto i : bdVetSet){
      if (!vetColor(i)) // only assign value to uncolored vet
        vetVal(i) = dis_int_1N4(gen);
      else
        vetVal(i) = -1; // if already colored, set random value to -1
    }

    if (verbose_1){
      MPI_Barrier(MPI_COMM_WORLD);
      cout << endl << "Rank: " << mpi_rank << " bd vet select random var finished!" << endl;
    }

    // async all to all send bdVet's assigned random values to other values
    vector<int> send_buffer_randVal;
    vector<int> recv_buffer_randVal(extVetNum.back() + extVetDisp.back());
    for (int pid = 0; pid < mpi_size; pid++) {
      if (!bdVetPid[pid].empty()){
        for (const auto& bdVet : bdVetPid[pid]){
          send_buffer_randVal.push_back(vetVal(bdVet)); // add the bd random val into the send buffer
        }
      }
    }

    if (verbose_1){
      MPI_Barrier(MPI_COMM_WORLD);
      cout << endl << "Rank: " << mpi_rank << " send_buffer composed !" << endl;
      for (int rank = 0; rank < mpi_size; rank++){
        MPI_Barrier(MPI_COMM_WORLD);
        if (mpi_rank == rank) {
          cout << "--------------------------" << endl;
          cout << "rank: " << mpi_rank << endl;
          cout << "sendSize: " << send_buffer_randVal.size() << endl;
          cout << "recvSize: " << recv_buffer_randVal.size() << endl;
          cout << endl << endl;
        }
      }
    }


    MPI_Request request_ext_recv;
//    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Ialltoallv(send_buffer_randVal.data(), bdVetNum.data(), bdVetDisp.data(), MPI_INT, recv_buffer_randVal.data(), extVetNum.data(), extVetDisp.data(), MPI_INT, MPI_COMM_WORLD, &request_ext_recv);


    // assign a random val to interior vet,  vetVal
    for (const auto i : inVetSet){
      if (!vetColor(i)) // only assign value to uncolored vet
        vetVal(i) = dis_int_1N4(gen);
      else if (vetVal(i) != -1)
        vetVal(i) = -1;
    }

    // find local minima first -> coloring interior part
    for (const auto &v_cur : inVetSet){
      if (!vetColor(v_cur)){ // if cur vet has not been colored (!0 = 1, !(int>0) = 0)
        bool isLocalMin = true;
        for (const auto &v_adj : adjTab[v_cur]){
          // only label not local min when 1. compared to non-colored adj 2.still possible for local min 3. found not local min
          if (isLocalMin && (!vetColor(v_adj)) && vetVal(v_cur) > vetVal(v_adj))
            isLocalMin = false;
        }
        if (isLocalMin){ // color this inVet!!!
          vetColor(v_cur) = cur_color; // current vet colored
          inVetThisColor.push_back(v_cur); // order does not matter, right?
        }
      }
    }

    if (verbose_1){
      MPI_Barrier(MPI_COMM_WORLD);
      cout << endl << "Rank: " << mpi_rank << " internal coloring finished!" << endl;
    }


    // wait until all the recv finished, coloring the boundary part
    MPI_Wait(&request_ext_recv, MPI_STATUS_IGNORE);

    unordered_map<int, int> extVetValTab; // store recv extVet val into table
    for (int i = 0; i < extVetPid1d.size(); i++){
      extVetValTab[extVetPid1d[i]] = recv_buffer_randVal[i];
    }

    if (verbose_1){
      for (int rank = 0; rank < mpi_size; rank++){
        MPI_Barrier(MPI_COMM_WORLD);
        if (mpi_rank == rank) {
          cout << "--------------------------" << endl;
          cout << "rank: " << mpi_rank << endl;
          cout << "bd val:" << endl;
          for (auto& bdVet : bdVetSet){
            cout << "bdVet: " << bdVet << "; Val: " << vetVal(bdVet) << endl;
          }

          cout << "ext val: " << endl;
          for (auto& extVet : extVetSet){
            cout << "extVet: " << extVet << "; Val: " << extVetValTab[extVet] << endl;
          }
          cout << endl << endl;
        }
      }
    }

    for (const auto &i : bdVetSet){
      int v_cur = i; // current bd vertex
      if (!vetColor(v_cur)) { // if current have not been colored
        bool isLocalMin = true;
        int val_cur = vetVal(v_cur);
        for (const auto &v_adj : adjTab[v_cur]){
          if (isLocalMin) {
            int val_adj;
            bool adjIsColored;
            if (extVetSet.count(v_adj)){ // if adjVet is extVet, fetch the value from table
              val_adj = extVetValTab[v_adj];
              adjIsColored = (val_adj == -1); // ext adjVet val would be -1 if it is colored
            } else { // if adj is internal vet, fetch the value directly
              val_adj = vetVal(v_adj);
              adjIsColored = (vetVal(v_adj) == -1); // -1 if colored, same for internal vet
            }
            // if the adj has not been colored and the adj val is smaller the cur -> cur is not local min
            if (!adjIsColored && val_adj < val_cur){
              isLocalMin = false;
            }
          }
        }
        if (isLocalMin){ // within the if of checking whether current have been colored or not
          vetColor(v_cur) = cur_color; // set the color
//          cout << cur_color << endl;
          bdVetThisColor.push_back(v_cur);
        }
      }
    }


    // get the num of vet num of current partition
    int colored_num = 0;
    for (const auto& i : vetColorVec){
      if (i)
        colored_num++;
    }

    // blocking mpi_allreduce, which enforce sync...
    colored_num_total_new = 0;
    MPI_Allreduce(&colored_num, &colored_num_total_new, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (verbose_3) {
      MPI_Barrier(MPI_COMM_WORLD);
//      stop = true;
//      if (cur_iter == 2) stop = true;
      if (mpi_rank == 0) {
        cout << "==================================" << endl << "Cur Iter: " << cur_iter << " ; Cur Color: " << cur_color << endl
             << "==================================" << endl << endl;
        cout << "colored_num_total_new: " << colored_num_total_new << "; colored_num_total_old: " << colored_num_total_old << endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
      for (int rank = 0; rank < mpi_size; rank++){
        MPI_Barrier(MPI_COMM_WORLD);
        if (mpi_rank == rank) {
          cout << "--------------------------" << endl;
          cout << "rank: " << mpi_rank << endl;
          cout << "vetColor!!! :" << endl;
          for (auto& vet : inVetSet){
            cout << "InVet: " << vet << ", color: " << vetColor(vet) << " , val: " << vetVal(vet) << endl;
          }
          for (auto& vet : bdVetSet){
            cout << "BdVet: " << vet << ", color: " << vetColor(vet) << " , val: " << vetVal(vet) << endl;
            if (vet == 8){
              cout << "Vet 8's adj: 3, val " << vetVal(3) << " , adj 7, val " << extVetValTab[7] << endl;
            }
          }
          cout << endl << endl;
        }
      }
    }

    double frac_cur = double(colored_num_total_new) / double(graph_serial_size);
    if (mpi_rank == 0){
//      cout << "colored_num_total is: " << colored_num_total << endl;
      cout << "frac_cur is: " << frac_cur << " / " << fracLim << endl;
    }

    if(colored_num_total_new != colored_num_total_old){ // add colored vets in to corresponding vectors
      colorInVet.push_back(inVetThisColor);
      colorBdVet.push_back(bdVetThisColor);
    }

    if (frac_cur >= fracLim) // if a large portion of vertex have been colored
    {
      stop = true;
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }


  // for debug, hardcode the color of first graph
  if (hardCode){
    cur_color = 4;
    if (mpi_rank == 0) {
      vetColor(0) = 2;
      vetColor(1) = 3;
      vetColor(2) = 2;
      vetColor(3) = 3;
      vetColor(4) = 4;
      vetColor(5) = 3;
      vetColor(6) = 4;
    }
    if (mpi_rank == 1){
      vetColor(7) = 3;
      vetColor(8) = 2;
      vetColor(9) = 1;
      vetColor(10) = 2;
      vetColor(11) = 1;
      vetColor(12) = 2;
      vetColor(13) = 3;
    }
    if (mpi_rank == 2){
      vetColor(14) = 2;
      vetColor(15) = 3;
      vetColor(16) = 2;
      vetColor(17) = 3;
      vetColor(18) = 1;
      vetColor(19) = 3;
    }
    // refresh the colorInVet and colorBdVet
    colorInVet.clear();
    colorBdVet.clear();
    colorBdVet.resize(cur_color);
    colorInVet.resize(cur_color);
    for (auto& inVet : inVetSet){
      colorInVet[vetColor(inVet) - 1].push_back(inVet);
    }
    for (auto& bdVet : bdVetSet){
      colorBdVet[vetColor(bdVet) - 1].push_back(bdVet);
    }
    hardCode = false; // only do it for the first time of coloring, used for correctness testing
  }



  // after coloring finish, build colorExtVet for each rank
  vector<int> buffer_send_bdColor(bdVetPid1d.size());
  vector<int> buffer_recv_extColor(extVetPid1d.size());
  for (int i = 0; i < bdVetPid1d.size(); i++){
    buffer_send_bdColor[i] = vetColor(bdVetPid1d[i]);
  }

  MPI_Alltoallv(buffer_send_bdColor.data(), bdVetNum.data(), bdVetDisp.data(), MPI_INT, buffer_recv_extColor.data(), extVetNum.data(), extVetDisp.data(), MPI_INT, MPI_COMM_WORLD);

  colorExtVet.resize(cur_color);
  for (int i = 0; i < extVetPid1d.size(); i++){
    colorExtVet[buffer_recv_extColor[i] - 1].push_back(extVetPid1d[i]);
    extVetColor[extVetPid1d[i]] = buffer_recv_extColor[i] - 1;
  }

  if(mpi_rank == 0) cout << endl << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl << "Coloring finished!!!!" << endl << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl << endl;



//  int i = 0;
//  while (!i)
//    sleep(5);

  // compose the communication vet array, disp array and num array for color_idx = 0, 2, ...., N-1
  // used to notify other rank about cur bdVet match result, before matching the next color HE
  // to avoid, during the next color's HEM stage, other rank's bdVet matches already matched result.
  // only need to do it for the bdVet

  vector<vector<int>> color_extVetPid1d(cur_color);
  vector<vector<set<int>>> color_extVetPid(cur_color, vector<set<int>>(mpi_size));
  vector<vector<int>> color_extVetNum(cur_color, vector<int>(mpi_size));
  vector<vector<int>> color_extVetDisp(cur_color, vector<int>(mpi_size));

  vector<vector<int>> color_bdVetPid1d(cur_color);
  vector<vector<set<int>>> color_bdVetPid(cur_color, vector<set<int>>(mpi_size));
  vector<vector<int>> color_bdVetNum(cur_color, vector<int>(mpi_size));
  vector<vector<int>> color_bdVetDisp(cur_color, vector<int>(mpi_size));

//  int i = 0;
//  while (!i)
//    sleep(5);

  for (auto& extVet : extVetSet){
    int pid = whichPid(extVet, vtxdist);
    int color = extVetColor[extVet];
    color_extVetPid[color][pid].insert(extVet);
    color_extVetNum[color][pid]++;
  }
  for (int color = 0; color < cur_color; color++){
    cur_disp = 0;
    for (int i = 0; i < mpi_size; i++){
      color_extVetDisp[color][i] = cur_disp;
      cur_disp += color_extVetNum[color][i];
    }
  }

  for (auto& bdVet : bdVetSet){
    for (auto& extVet : adjTab[bdVet]){
      int color = vetColor(bdVet) - 1;
      int pid = whichPid(extVet, vtxdist);
      if (pid != mpi_rank){
        color_bdVetPid[color][pid].insert(bdVet);
//        bdVetNum[pid]++; // oops, leads to bug! non-duplicated!!!!
      }
    }
  }

  for (int color = 0; color < cur_color; color++){
    for (int pid = 0; pid < mpi_size; pid++){
      color_bdVetNum[color][pid] = color_bdVetPid[color][pid].size();
    }
  }

  for (int color = 0; color < cur_color; color++){
    cur_disp = 0;
    for (int i = 0; i < mpi_size; i++){
      color_bdVetDisp[color][i] = cur_disp;
      cur_disp += color_bdVetNum[color][i];
    }
  }


  for (int color = 0; color < cur_color; color++){
    for (auto& set_pid : color_extVetPid[color]){
      for (auto& extVet : set_pid){
        color_extVetPid1d[color].push_back(extVet);
      }
    }
    for (auto& set_pid : color_bdVetPid[color]){
      for (auto& bdVet : set_pid){
        color_bdVetPid1d[color].push_back(bdVet);
      }
    }
  }

//  int i = 0;
//  while (!i)
//    sleep(5);


  /* Graph Coarsening - 1 stage */


  if(mpi_rank == 0) cout << "start coarsening" << endl;



  unordered_map<int, int> match;
  for (auto& vetAdjPair : adjTab){
    match[vetAdjPair.first] = -1;
  }

  for (auto& extVet : extVetSet){
    match[extVet] = -1; // so match also have all the extVet's value...
  }

//  // cal the number of external edges
//  int numExtEdge = 0;
//  for (auto& pair : adjTab){
//    int v_cur = pair.first;
//    for (auto& v_adj : pair.second){
//      if (whichPid(v_adj, vtxdist) != mpi_rank){
//        numExtEdge ++;
//      }
//    }
//  }

//  if (verbose_3) cur_color = 1;// for testing, set to one

  map<int, int> giveTab; // cur rank bdVet(first) will be given away, extVet(second) will get this
  map<int, int> getTab; // cur rank bdVet(second) will get the extVet(first), store reversely due to further usage
  set<int> extGiveTab;

  for (int color_idx = 0; color_idx < cur_color; color_idx++){ // for each color
    /* Graph Matching using HEM */
    // writes structure for local
    unordered_map<int, vector<int>> local_write_request;

    // write structure for mpi
    vector<vector<writeMatch>> pidExtVetWrite(mpi_size); // used for send
    vector<int> writeNum_send(mpi_size);

    // the maximum size is the total number of external vet, at worst case, each extVet has only one HE for bdVet of current pid
    // IMPORTANT: the first elem of each pid's stripe is magic struct contains the stripe size -> max size += mpi_size
    vector<writeMatch> recv_buffer_write(extVetSet.size() + mpi_size);
    vector<int> writeDisp_recv(mpi_size);
    vector<int> writeNumMax_recv(mpi_size);
    for (int i = 0; i < mpi_size; i++){
      writeDisp_recv[i] = extVetDisp[i] + i; // due to magic struct showing array size
      writeNumMax_recv[i] = extVetNum[i] + 1; // due to magic struct ...
    }

    writeMatch magic{MAGIC_NUM, 0};
    for (int i = 0; i < mpi_size; i ++){
      pidExtVetWrite[i].push_back(magic);
    }

    // read structure for mpi

    // bd vet HEM trial, add into internal writeTab and external writeTab
    for (auto& v_cur : colorBdVet[color_idx]){ // for each bd vet!!!

      if (match[v_cur] == -1 && !adjTab[v_cur].empty()){ // if v_cur is unmatched and v_cur has v_adj
        // find the HE
        int v_matched = findHE(v_cur, adjTab, edgeWgtTab, match);

        if (extVetSet.count(v_matched)){ // if is the external, add into extWrite
          writeMatch myWriteMatch{v_matched, v_cur};
          int pid_to = whichPid(v_matched, vtxdist);
//          cout << "pid_to is: " << pid_to << endl;
          pidExtVetWrite[pid_to].push_back(myWriteMatch);
          pidExtVetWrite[pid_to][0].v_writor++; // modify the magic head!!!!
        } else { // if HE is internal... add into local write request
          local_write_request[v_matched].push_back(v_cur);
        }
      }
    }

    MPI_Request request_write[mpi_size];
    // non-blocking gather write request
    for (int pid = 0; pid < mpi_size; pid++){
      MPI_Igatherv(pidExtVetWrite[pid].data(), pidExtVetWrite[pid][0].v_writor + 1, mpi_write_type,
                   recv_buffer_write.data(), writeNumMax_recv.data(), writeDisp_recv.data(), mpi_write_type,
                   pid, MPI_COMM_WORLD, &request_write[pid]);
    }

//    int i = 0;
//    while (!i)
//      sleep(5);

    // inVet-inVet edge compare && label
    for (auto& v_cur : colorInVet[color_idx]){
      if (match[v_cur] == -1 && !adjTab[v_cur].empty()){
        int v_matched = findHE(v_cur, adjTab, edgeWgtTab, match);
        local_write_request[v_matched].push_back(v_cur); // all push into the write request
      }
    }


    // but only process inVet-inVet edge competition right now, leave the inVet-bdVet edge case
    for (auto& cur_pair : local_write_request){
      int v_matched = cur_pair.first;
      if (inVetSet.count(v_matched)){ // if v_matched is within internal
        set<pair<int, int>> HEM_ranking;
        for (auto& v_comp : cur_pair.second){
          pair v_comp_wgt(edgeWgtTab[key(v_matched, v_comp)], v_comp);
          HEM_ranking.insert(v_comp_wgt);
        }
        int v_winner = HEM_ranking.rbegin()->second; // you win!
        // only label the matched edge, each other, other still be -1, no change!
//        cout << "winner are <<: " << v_winner << endl;
        match[v_winner] = v_matched;
        match[v_matched] = v_winner;
      }
    }


    // blocking write recv request, based on write request, label the bdVet-to-inVet and bdVet-to-extVet HEM
    MPI_Wait(&request_write[mpi_rank], MPI_STATUS_IGNORE); // only need to wait until the current is finished


//    if (verbose_3){ // show the match
//      for (int rank = 0; rank < mpi_size; rank++){
//        MPI_Barrier(MPI_COMM_WORLD);
//        if (mpi_rank == rank) {
//          cout << "--------------------------" << endl;
//          cout << "rank: " << mpi_rank << endl;
//          cout << "--------------------------" << endl;
//          cout << "Match info is: " << endl;
//          for (auto& pair : match){
//            cout << "vet: " << pair.first << " value is: " << match[pair.first] << endl;
//          }
//        }
//      }
//    }


    // unboxing write buffer
    vector<vector<writeMatch>> write_req_recv;
    write_req_recv.resize(mpi_size);
    for (int pid = 0; pid < mpi_size; pid++){
      int disp = writeDisp_recv[pid];
      int num = recv_buffer_write[disp].v_writor; // magic!
      if (recv_buffer_write[disp].v_to_write != MAGIC_NUM){
        cout << "unboxing failure!!!!!" << endl;
        exit(1111111);
      }
      for (int i = 0; i < num; i++){
        int cur_idx = disp + i + 1; // remember to: 1. skip first element for each rank; 2. while considering its effect on the disp...
        write_req_recv[whichPid(recv_buffer_write[cur_idx].v_writor, vtxdist)].push_back(recv_buffer_write[cur_idx]);
        int to_write = recv_buffer_write[cur_idx].v_to_write;
        int writor = recv_buffer_write[cur_idx].v_writor;
        if (match[to_write] == -1)
          local_write_request[to_write].push_back(writor); // add into local write, only when to_write is unmatched
      }
    }
    for (auto& cur_pair : local_write_request){
      int v_matched = cur_pair.first;
      if (match[v_matched] == -1 && bdVetSet.count(v_matched)){ // check for bdVet write request, should work for both bdVet-inVet and bdVet-extVet edges
        set<pair<int, int>> HEM_ranking;
        for (auto& v_comp : cur_pair.second){
          pair v_comp_wgt(edgeWgtTab[key(v_matched, v_comp)], v_comp);
          HEM_ranking.insert(v_comp_wgt);
        }
        int v_winner = HEM_ranking.rbegin()->second; // you win!
        // only label the matched edge, each other, other still be -1, no change!
//        cout << "winner are <<: " << v_winner << endl;
        match[v_winner] = v_matched; // might add extVet into the match key value...
        match[v_matched] = v_winner;
      }
    }


//    cout << " you even successfully reached here!" << endl;
//    int i = 0;
//    while (!i)
//      sleep(5);

    // prepare read, read also need to be gathered and send to specific position
    vector<int> readNum_recv(mpi_size);
    vector<int> readDisp_recv(mpi_size);
    vector<int> readNum_send(mpi_size);
    int disp_cur = 0;
    for (int pid = 0; pid < mpi_size; pid++){
      int size = pidExtVetWrite[pid].size() - 1;
      readNum_recv[pid] = size;
      readDisp_recv[pid] = disp_cur;
      disp_cur += size;
      readNum_send[pid] = recv_buffer_write[writeDisp_recv[pid]].v_writor; // get that from magic
    }

    vector<vector<readMatch>> pidRead_send;
    pidRead_send.resize(mpi_size);
    vector<readMatch> recv_buffer_read;
    // loop through received write request
    for (int pid = 0; pid < mpi_size; pid ++){
      for (auto& writeReq : write_req_recv[pid]){
        int v_to_write = writeReq.v_to_write;
        int v_writor = writeReq.v_writor;
        readMatch myRead{v_to_write, v_writor, match[v_to_write], -1};
        if (match[v_to_write] == v_writor){ // if write success, random choose side
//          cout << "bd-ext vet HE match between: bd: " << v_writor << " and ext: " << v_to_write << endl;
          bool randomBool = dis_int_01(gen);
          if (randomBool){// here writor is extVet, v_to_write is bdVet
            myRead.who_keeps = v_writor;
            giveTab[v_to_write] = v_writor; // oops, lose the v_to_write to the writor
          } else {
            myRead.who_keeps = v_to_write;
            getTab[v_writor] = v_to_write; // luckyly, here keep the v_to_write
          }
        }
        pidRead_send[pid].push_back(myRead);
      }
    }
    // non-blocking send (gather) back
    MPI_Request request_read[mpi_size];

    recv_buffer_read.resize(readDisp_recv.back() + readNum_recv.back());
    // non-blocking gather write request
    for (int pid = 0; pid < mpi_size; pid++){
      MPI_Igatherv(pidRead_send[pid].data(), readNum_send[pid], mpi_read_type,
                   recv_buffer_read.data(), readNum_recv.data(), readDisp_recv.data(), mpi_read_type,
                   pid, MPI_COMM_WORLD, &request_read[pid]);
    }

//    cout << " you even successfully reached here!" << endl;
//    int i = 0;
//    while (!i)
//      sleep(5);

    MPI_Wait(&request_read[mpi_rank], MPI_STATUS_IGNORE); // wait until the current pid recv read

//    if(color_idx == 1){
//      int i = 0;
//      while (!i)
//        sleep(5);
//    }

    // read bd vet HEM result && which side store matched Vet,
    // here, v_reader is bdVet, v_to_read is extVet
    for (int pid = 0; pid < mpi_size; pid++){
      int disp = readDisp_recv[pid];
      int num = readNum_recv[pid];
      for (int i = 0; i < num; i++){ // loop through each readMatch
        int cur_idx = disp + i;
        if (recv_buffer_read[cur_idx].value == recv_buffer_read[cur_idx].v_reader) { // reader is also writor, if match is successful!
          // modify the match of the bdVet
          match[recv_buffer_read[cur_idx].v_reader] = recv_buffer_read[cur_idx].v_to_read;
          match[recv_buffer_read[cur_idx].v_to_read] = recv_buffer_read[cur_idx].v_reader;
          // keep or give away?
          if (recv_buffer_read[cur_idx].who_keeps == recv_buffer_read[cur_idx].v_reader){ // bdVet in this rank get it!
            getTab[recv_buffer_read[cur_idx].v_to_read] = recv_buffer_read[cur_idx].v_reader;
          } else if (recv_buffer_read[cur_idx].who_keeps == recv_buffer_read[cur_idx].v_to_read) { // we loose it
            giveTab[recv_buffer_read[cur_idx].v_reader] = recv_buffer_read[cur_idx].v_to_read;
          } else {
           // raise error
           exit(2333);
          }
        } else { // if match failed, lost the competition....
          match[recv_buffer_read[cur_idx].v_reader] = -1; // change back to -1, may be duplicated...
          match[recv_buffer_read[cur_idx].v_to_read] = recv_buffer_read[cur_idx].value; // to_write robbed by...
        }
      }
    }

    // update the match result of extVet as well as the give or get info (for bdVet connect to multiple rank...)
    vector<matchRes> bdVetMatch_send; // (color_bdVetNum[color_idx].back() + color_bdVetDisp[color_idx].back());
    vector<matchRes> extVetMatch_recv(color_extVetNum[color_idx].back() + color_extVetDisp[color_idx].back());

    for (auto& bdVet : color_bdVetPid1d[color_idx]){
      matchRes myRes{match[bdVet], 0};
      if (giveTab.count(bdVet) != 0){
        myRes.isGiven = 1;
      }
      bdVetMatch_send.push_back(myRes);
    }


    MPI_Alltoallv(bdVetMatch_send.data(), color_bdVetNum[color_idx].data(), color_bdVetDisp[color_idx].data(), mpi_match_res_type,
                  extVetMatch_recv.data(), color_extVetNum[color_idx].data(), color_extVetDisp[color_idx].data(), mpi_match_res_type, MPI_COMM_WORLD);


    for (int i = 0; i < color_extVetPid1d[color_idx].size(); i++){
      int extVet = color_extVetPid1d[color_idx][i];
      int match_val = extVetMatch_recv[i].val;
      match[extVet] = match_val; // update the match value!!
      if (extVetMatch_recv[i].isGiven){
        extGiveTab.insert(extVet);
      }
    }

//    cout << " you even successfully reached here!" << endl;
//    if (color_idx == 2){
//      int i = 0;
//      while (!i)
//        sleep(5);
//    }


    if (verbose_3){ // show the match
      for (int rank = 0; rank < mpi_size; rank++){
        MPI_Barrier(MPI_COMM_WORLD);
        if (mpi_rank == rank) {
          cout << "--------------------------" << endl;
          cout << "rank: " << mpi_rank << endl;
          cout << "--------------------------" << endl;
          cout << "Match info is: " << endl;
          for (auto& pair : match){
            if (!extVetSet.count(pair.first))
              cout << "in/bdVet: " << pair.first << " value is: " << match[pair.first] << endl;
          }
          for (auto& pair : match){
            if (extVetSet.count(pair.first))
              cout << "extVet: " << pair.first << " value is: " << match[pair.first] << endl;
          }
        }
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
//    if (mpi_rank == 2 && color_idx == 1){cout << "after iter 3, match [8] is: " << match[8] << endl;}
  }

  // sequential printing
  if (verbose_4) {
    for (int rank = 0; rank < mpi_size; rank++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (mpi_rank == rank) {
        cout << endl << "rank: " << mpi_rank << endl;
        cout << endl << "getTab:" << endl;
        for (auto &val: getTab) {
          cout << "Vet: " << val.second << "; get the Vet: " << val.first << endl;
        }
        cout << endl << "giveTab:" << endl;

        for (auto &val: giveTab) {
          cout << "Vet: " << val.first << "; is given to Vet: " << val.second << endl;
        }
        cout << endl << endl;
      }
    }
  }


  // reorganize the adjTab .....
  if(mpi_rank == 0) cout << endl << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl << "Coarsening finished!!!!" << endl << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << endl << endl;

  //here, for simplicity, we assume that smaller vetIdx hold the subgraph except cross-rank match

  // need to modify the ext-bd connection locally, simply using match[extVet],
  // for bdVet, for its v_adj,
  //     if v_adj is matched
  //        get v_get, v_lose
  //        if v_adj in other rank
  //           if v_adj != v_get
  //            add v_get into bdVet adjList,
  //            erase v_lose from bdVet adjList
  //            bdVet-v_get += bdVet-v_adj.
  //            erase bdVet-v_adj
  //           else
  //            do nothing
  //        else (v_get and v_lose in cur rank) so, leve this to next procs
  //          do nothing

//  dumpAll
//
//  int i = 0;
//  while (!i)
//    sleep(5);

  for (auto& bdVet : bdVetSet){
    auto copy = adjTab[bdVet];
    for (auto& v_adj : copy){
      if (adjTab.count(v_adj) == 0 && match[v_adj] != -1 ){
        int v_get;
        int v_lose;
        if (extGiveTab.count(v_adj)){
          v_get = match[v_adj];
          v_lose = v_adj;
        } else {
          v_get = min(v_adj, match[v_adj]);
          v_lose = max(v_adj, match[v_adj]);
        }

        if (v_adj != v_get && v_get != bdVet && whichPid(v_get, vtxdist) != mpi_rank){ // need to adjust
//          if (bdVet == 15) exit(15);
          adjTab[bdVet].insert(v_get);
          adjTab[bdVet].erase(v_lose);
//          if (bdVet == 6 && v_get == 2) {
//            cout << "test" << edgeWgtTab[key(bdVet, v_adj)] << endl;
//            exit(3);
//          }
          edgeWgtTab[key(bdVet, v_get)] += edgeWgtTab[key(bdVet, v_adj)];
          edgeWgtTab.erase(key(bdVet, v_adj));
        }
      }
    }
  }

//    dumpAll
//  int i = 0;
//  while (!i)
//    sleep(5);



  // for no cross rank in/bd-in/bd match
  for (auto& pair : match){
    int vet = pair.first;
    int val = pair.second;
    // if has a match + is not external-in/bd edge
    if (val != -1 && adjTab.count(vet) != 0 && adjTab.count(val) != 0){
      int v_get = min(vet, val);
      int v_lose = max(vet, val);
      if (adjTab.count(v_lose) != 0){ // prevent do it twice
        auto copy = adjTab[v_lose];
        for (auto v_adj : copy){
          if (v_adj != v_get){
            adjTab[v_get].insert(v_adj);
            adjTab[v_lose].erase(v_adj);
            edgeWgtTab[key(v_get, v_adj)] += edgeWgtTab[key(v_lose, v_adj)];
//            if (v_get == 14 && v_adj == 8){
//              cout << mpi_rank << endl;
//              exit(10);
//            }
            edgeWgtTab.erase(key(v_lose, v_adj));
            if (adjTab.count(v_adj) != 0){
              adjTab[v_adj].insert(v_get);
              adjTab[v_adj].erase(v_lose);
//              if (v_get == 1 && v_adj == 2){
//                cout << edgeWgtTab[key(v_adj, v_get)] << ", " << edgeWgtTab[key(v_adj, v_lose)] << endl;
//                cout << "v_lose" << v_lose;
//                exit(10);
//              }
              edgeWgtTab[key(v_adj, v_get)] += edgeWgtTab[key(v_adj, v_lose)];

              edgeWgtTab.erase(key(v_adj, v_lose));
            }
          }
        }
        adjTab.erase(v_lose);
        adjTab[v_get].erase(v_lose);
        edgeWgtTab.erase(key(v_get, v_lose));
        edgeWgtTab.erase(key(v_lose, v_get));
      }
    }
  }
//
//  dumpAll
//  int i = 0;
//  while (!i)
//    sleep(5);


  // sequential printing
  if (verbose_3) {
    for (int rank = 0; rank < mpi_size; rank++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (mpi_rank == rank) {
        cout << endl << "rank: " << mpi_rank << endl;
        cout << endl << "adj info:" << endl;
        for (auto &val: adjTab) {
          cout << "vet: " << val.first << " is adj to: " << endl;
          for (auto &adjVet: val.second) {
            cout << adjVet << ", ";
          }
          cout << endl;
        }
        cout << endl << "wgt info:" << endl;

        for (auto &val: edgeWgtTab) {
          cout << "edge: " << key2a(val.first) << "-" << key2b(val.first) << ": " << val.second << endl;
        }
        cout << endl;

      }
    }
  }

//  int i = 0;
//  while (!i)
//    sleep(5);


  // bdVet that is losing, sending adjlist size
  // bdVet that is getting, recv adjList size
  vector<set<int>> pidBdVet_lose(mpi_size);
  vector<set<int>> pidExtVet_get(mpi_size);
  vector<int> pidBdVet_lose1d;
  vector<int> pidExtVet_get1d;
  vector<int> pidBdVet_loseNum(mpi_size);
  vector<int> pidExtVet_getNum(mpi_size);
  vector<int> pidBdVet_loseDisp(mpi_size);
  vector<int> pidExtVet_getDisp(mpi_size);


  for (auto& pair : giveTab){
    int bdVet_lose = pair.first;
    int extVet_get = pair.second;
    int pid = whichPid(extVet_get, vtxdist);

    pidBdVet_lose[pid].insert(bdVet_lose);
    pidBdVet_loseNum[pid]++;
  }

  for (auto& pair : getTab){
    int bdVet_get = pair.second;
    int extVet_lose = pair.first;
    int pid = whichPid(extVet_lose, vtxdist);

    pidExtVet_get[pid].insert(extVet_lose);
    pidExtVet_getNum[pid]++;
  }

  // unroll to 1d
  for (auto& set : pidBdVet_lose){
    for (auto bdVet : set){
      pidBdVet_lose1d.push_back(bdVet);
    }
  }
  for (auto& set : pidExtVet_get){
    for (auto extVet : set){
      pidExtVet_get1d.push_back(extVet);
    }
  }

  int cur_disp_1 = 0;
  int cur_disp_2 = 0;
  for (int i = 0; i < mpi_size; i++){
    pidBdVet_loseDisp[i] = cur_disp_1;
    pidExtVet_getDisp[i] = cur_disp_2;
    cur_disp_1 += pidBdVet_loseNum[i];
    cur_disp_2 += pidExtVet_getNum[i];
  }

  vector<int> bdLoseAdjSize;
  vector<int> extGetAdjSize(pidExtVet_getNum.back() + pidExtVet_getDisp.back());
  for (auto& bdVet : pidBdVet_lose1d){
    bdLoseAdjSize.push_back(adjTab[bdVet].size());
  }

//  int i = 0;
//  while (!i)
//    sleep(5);

  MPI_Request request_send_num;
  MPI_Ialltoallv(bdLoseAdjSize.data(), pidBdVet_loseNum.data(), pidBdVet_loseDisp.data(), MPI_INT,
                extGetAdjSize.data(), pidExtVet_getNum.data(), pidExtVet_getDisp.data(), MPI_INT, MPI_COMM_WORLD, &request_send_num);

//  int i = 0;
//  while (!i)
//    sleep(5);

  // allocate the buffer to actually hold the adjList size
  vector<vetWgt> buffer_bdAdj_send;
  vector<vetWgt> buffer_extAdj_recv;

  vector<int> adj_info_disp_send(mpi_size);
  vector<int> adj_info_disp_recv(mpi_size);

  vector<int> adj_info_num_send(mpi_size); // for each pid, sum(v_adj of extVetGet)
  vector<int> adj_info_num_recv(mpi_size);

  // collect the send buffer
  for (int pid = 0; pid < mpi_size; pid++){
    for (auto& bdVet_send : pidBdVet_lose[pid]){
      for (auto& v_adj : adjTab[bdVet_send]){
        vetWgt v_adj_wgt{v_adj, edgeWgtTab[key(bdVet_send, v_adj)]};
        buffer_bdAdj_send.push_back(v_adj_wgt);
      }
    }
  }

  // build the send buffer num and disp
  int total_disp = 0;
  for (int pid = 0; pid < mpi_size; pid++){ // loop through each rank
    int numAdj = 0;
    int disp = pidBdVet_loseDisp[pid];
    for (int i = 0; i < pidBdVet_loseNum[pid]; i++) {
      numAdj += bdLoseAdjSize[disp + i];
    }
    adj_info_num_send[pid] = numAdj;
    adj_info_disp_send[pid] = total_disp;
    total_disp +=  numAdj;
  }

  // to overlap, could move the bdVet-extVet giving case modification here ... do it later...


//  int i = 0;
//  while (!i)
//    sleep(5);

  // build the recv buffer num and disp
  MPI_Wait(&request_send_num, MPI_STATUS_IGNORE);
//  cout << "You get here 1 !!" << endl;
  total_disp = 0;
  for (int pid = 0; pid < mpi_size; pid++){
    int numAdj = 0;
    int disp = pidExtVet_getDisp[pid];
    for (int i = 0; i < pidExtVet_getNum[pid]; i++) {
      numAdj += extGetAdjSize[disp + i];
    }
    adj_info_num_recv[pid] = numAdj;
    adj_info_disp_recv[pid] = total_disp;
    total_disp +=  numAdj;
  }

  buffer_extAdj_recv.resize(adj_info_num_recv.back() + adj_info_disp_recv.back());

//  int i = 0;
//  while (!i)
//    sleep(5);

  MPI_Alltoallv(buffer_bdAdj_send.data(), adj_info_num_send.data(), adj_info_disp_send.data(), mpi_vetWgt_type,
                buffer_extAdj_recv.data(), adj_info_num_recv.data(), adj_info_disp_recv.data(), mpi_vetWgt_type, MPI_COMM_WORLD);

  // for each bdVet-extVet match && give it to other rank, do the following
  // for bdVet's v_adj:
  // if v_adj == extVet, del bdVet-extVet wgt,
  // else if v_adj is not external
  //    if v_adj is not given away
  //    add extVet to bdVet's v_adj list,
  //    erase bdVet from v_adj list
  //    increase v_adj-extVet wgt by bdVet-v_adj wgt, erase bdVet-v_adj and v_adj-bdVet
  // else (v_adj is external and v_adj != extVet)
  //    erase bdVet-v_adj
  // Finally, delete adjTab[bdVet],

  // Q: if extVet is accidentally given back / or to other rank?
  // Ans: Just let the getTab algorithm take care of it. Because we do not know extVet will go to where....
  //
  for (auto& pair : giveTab){ // directly use the give tab
    int bdVet = pair.first; // given to extVet
    int extVet = pair.second;
    for (auto v_adj : adjTab[bdVet]){
      if (v_adj == extVet){
        edgeWgtTab.erase(key(bdVet,extVet));
      }
      else if (adjTab.count(v_adj) != 0){ // not external
        adjTab[v_adj].insert(extVet);
        adjTab[v_adj].erase(bdVet);
        edgeWgtTab[key(v_adj, extVet)] += edgeWgtTab[key(bdVet, v_adj)];
        edgeWgtTab.erase(key(bdVet, v_adj));
        edgeWgtTab.erase(key(v_adj, bdVet));
      } else { // is external
        edgeWgtTab.erase(key(bdVet, v_adj));
      }
    }
    adjTab.erase(bdVet);
  }

//  dumpAll
//  int i = 0;
//  while (!i)
//    sleep(5);

  // for each bdVet-extVet match && get, do the following,
  // for extVet's v_adj:
  // if v_adj == extVet, del erase extVet from bdVet adjList, del bdVet-extVet wgt
  // else if v_adj is not external
  //    add v_adj into bdVet adjList
  //    erase extVet from v_adj's adjList
  //    increase v_adj-bdVet and bdVet-v_adj by v_adj-extVet (sent)
  //    erase v_adj-extVet
  // else (if v_adj is external and v_adj != extVet)
  //    add v_adj into bdVet adjList
  //    increase bdVet-v_adj by v_adj-extVet (sent)
  //

  int cur_adj_disp = 0;
  for (int i = 0; i < pidExtVet_get1d.size(); i++){
    int extVet = pidExtVet_get1d[i];
    int bdVet = getTab[extVet]; // that's why we store getTab using extVet as key value....
    int num_adj = extGetAdjSize[i];

    if (extVet == 7){
      for (int d = cur_adj_disp; d < (cur_adj_disp + num_adj); d++)
        cout << "test " << buffer_extAdj_recv[d].wgt << ", ";
    }

    for (int d = cur_adj_disp; d < (cur_adj_disp + num_adj); d++){
      int v_adj = buffer_extAdj_recv[d].v_adj;
      int wgt = buffer_extAdj_recv[d].wgt; // which is extVet-v_adj...
      if (v_adj == bdVet){
        adjTab[bdVet].erase(extVet);
        edgeWgtTab.erase(key(bdVet,extVet));
      }
      else if (adjTab.count(v_adj) != 0){ // not external
        adjTab[bdVet].insert(v_adj);
        adjTab[v_adj].insert(bdVet);
        adjTab[v_adj].erase(extVet);
        edgeWgtTab[key(v_adj, bdVet)] += wgt;
        edgeWgtTab[key(bdVet, v_adj)] += wgt;
        edgeWgtTab.erase(key(v_adj, extVet));
      }
      else if (extGiveTab.count(v_adj)){ // if the external is given out randomly
        int v_get_adj = match[v_adj];
        adjTab[bdVet].erase(v_adj);
        adjTab[bdVet].insert(v_get_adj);
        edgeWgtTab[key(bdVet, v_get_adj)] += wgt;
        edgeWgtTab.erase(key(v_adj, extVet));
      }
      else
      {  // if external and except extVet and is not given out...
        adjTab[bdVet].insert(v_adj);
        edgeWgtTab[key(bdVet, v_adj)] += wgt;
      }
    }

    cur_adj_disp += num_adj;
  }



//  // sequential printing
  dumpAll
//  int i = 0;
//  while (!i)
//    sleep(5);

//  int i = 0;
//  while (!i)
//    sleep(5);









  MPI_Finalize();
  return 0;
}
