#include<bits/stdc++.h>
// #include<omp.h>
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
// #include <tbb/blocked_range.h>

using namespace std;

const int NUMTHEADS = 8;
int Kmax = 0;

struct Graph{
    int n, m;

    vector<pair<int, int> >edges;
    set<int> nodes;
    unordered_map<int, int> mp;
    vector<unordered_set<int> > N;
    vector<int> ID;

    Graph() : n(0), m(0) {}

    void init(){
        freopen("1.txt","r",stdin);
        int u, v;
        while(scanf("%d%d", &u, &v) != EOF){
            if(u == v) continue;
            nodes.insert(u);
            nodes.insert(v);
            edges.push_back({u, v});
        }
        ID.resize(nodes.size() + 1);
        for(auto &x: nodes){
            ID[++n] = x;
            mp[x] = n;
        }
        m = edges.size();
        N.resize(n + 1);
        for(auto &e: edges){
            int u = mp[e.first];
            int v = mp[e.second];
            N[u].insert(v);
            N[v].insert(u);
        }
        fclose(stdin);
    }
};

// inline vector<int> Union(vector<int> &A, vector<int> &B){
//     vector<int> ret;
//     int n = A.size();
//     int m = B.size();
//     int i = 0;
//     int j = 0;
//     while(i < n && j < m){
//         if(A[i] == B[j]){
//             ret.push_back(A[i]);
//             ++i;
//             ++j;
//         }
//         else if(A[i] < B[j]){
//             ret.push_back(A[i]);
//             ++i;
//         }
//         else {
//             ret.push_back(B[j]);
//             ++j;
//         }
//     }
//     while(i < n){
//         ret.push_back(A[i]);
//         ++i;
//     }
//     while(j < m){
//         ret.push_back(B[j]);
//         ++j;
//     }
//     return ret;
// }

// inline vector<int> Intersection(vector<int> &A, vector<int> &B){
//     vector<int> ret;
//     int n = A.size();
//     int m = B.size();
//     int i = 0;
//     int j = 0;
//     while(i < n && j < m){
//         if(A[i] == B[j]){
//             ret.push_back(A[i]);
//             ++i;
//             ++j;
//         }
//         else if(A[i] < B[j]){
//             ++i;
//         }
//         else {
//             ++j;
//         }
//     }
//     return ret;
// }

inline unordered_set<int> Union(const unordered_set<int> &A, const unordered_set<int> &B){
    unordered_set<int> ret = A;
    for(auto &x: B){
        ret.insert(x);
    }
    return ret;
}

inline unordered_set<int> Intersection(const unordered_set<int> &A, const unordered_set<int> &B){
    unordered_set<int> ret;
    if(A.size() < B.size()){
        for(auto &x: A){
            if(B.find(x) != B.end()){
                ret.insert(x);
            }
        }
    }
    else{
        for(auto &x: B){
            if(A.find(x) != A.end()){
                ret.insert(x);
            }
        }
    }
    return ret;
}

inline int IntersectionSize(const unordered_set<int> &A, const unordered_set<int> &B){
    int ret = 0;
    if(A.size() < B.size()){
        for(auto &x: A){
            if(B.find(x) != B.end()){
                ++ret;
            }
        }
    }
    else{
        for(auto &x: B){
            if(A.find(x) != A.end()){
                ++ret;
            }
        }
    }
    return ret;
}

vector<vector<int> > maximal_cliques;
vector<vector<vector<int> *>> k_cliques;
vector<vector<int> *> all_cliques;
vector<int> rows_to_consider;

inline int getPivot(const unordered_set<int> &P, const unordered_set<int> &X, Graph const *G){
    int v = 0;
    int mx = -1;
    for(auto &x: P){
        int t = IntersectionSize(P, G -> N[x]);
        if(t > mx){
            v = x;
            mx = t;
        }
    }
    for(auto &x: X){
        if(P.find(x) == P.end()) continue;
        int t = IntersectionSize(P, G -> N[x]);
        if(t > mx){
            v = x;
            mx = t;
        }
    }
    return v;
}

void BKPivot(unordered_set<int> &R, unordered_set<int> &P, unordered_set<int> &X, Graph const *G){
    if(!P.size()){
        if(!X.size()){
            vector<int>tmp;
            for(auto &x: R){
                tmp.push_back(x);
            }
            sort(tmp.begin(), tmp.end());
            maximal_cliques.emplace_back(tmp);
        }
        return ;
    }
    int t = getPivot(P, X, G);
    vector<int>candidates;
    for(auto &x: P){
        if((G -> N[t]).find(x) != (G -> N[t]).end()) continue;
        candidates.push_back(x);
    }    
    for(auto &x: candidates){
        if((G -> N[t]).find(x) != (G -> N[t]).end()) continue;
        auto RR = R; RR.insert(x);
        auto PP = Intersection(P, G -> N[x]);
        auto XX = Intersection(X, G -> N[x]);
        BKPivot(RR, PP, XX, G);
        P.erase(x);
        X.insert(x);
    }
}

void FindMaximalCliques(Graph *G){
    unordered_set<int> R;
    unordered_set<int> P;
    unordered_set<int> X;
    for(int i = 1; i <= (G -> n); ++i){
        P.insert(i);
    }
    BKPivot(R, P, X, G);

    // freopen("3.txt","r",stdin);
    // int x;
    // vector<int> tmp;
    // while(scanf("%d", &x) != EOF){
    //     if(x == -1){
    //         maximal_cliques.emplace_back(tmp);
    //         tmp.clear();
    //     }
    //     else{
    //         tmp.push_back(G -> mp[x]);
    //     }
    // }
    // fclose(stdin);

    for(auto &clique: maximal_cliques){
        if(clique.size() > Kmax){
            Kmax = clique.size();
        }
    }
    k_cliques.resize(Kmax - 1);
    rows_to_consider.resize(Kmax - 1);
    for(auto &clique: maximal_cliques){
        k_cliques[clique.size() - 2].emplace_back(&clique);
    }
    printf("total maximal cliques : %d\n", maximal_cliques.size());
    printf("maximal clique size: %d\n", Kmax);
    int num = 0;
    for(int k = Kmax; k >= 2; --k){
        // printf("%d-clique : %d\n", k, k_cliques[k - 2].size());
        num += k_cliques[k - 2].size();
        rows_to_consider[k - 2] = num;
        for(auto &clique: k_cliques[k - 2]){
            all_cliques.emplace_back(clique);
            // for(auto &x: *clique){
            //     printf("%d ", G -> ID[x]);
            // }
            // puts("");
        }
    }
}

struct UnionFind{
    vector<int> fa;
    UnionFind(int n) : fa(n){
        for(int i = 0; i < n; ++i){
            fa[i] = i;
        }
    }

    int Find(int x){
        assert(x < fa.size());
        return x == fa[x] ? x : fa[x] = Find(fa[x]);
    }

    void Union(int u, int v){
        assert(u < fa.size());
        assert(v < fa.size());
        fa[Find(u)] = Find(v);
    }
};

inline int Overlap(const vector<int> &A, const vector<int> &B){
    int ret = 0;
    int n = A.size();
    int m = B.size();
    int i = 0;
    int j = 0;
    while(i < n && j < m){
        if(A[i] == B[j]){
            ++ret;
            ++i;
            ++j;
        }
        else if(A[i] < B[j]){
            ++i;
        }
        else {
            ++j;
        }
    }
    return ret;
}


vector<UnionFind *> F[NUMTHEADS];
void COSpoc_work(int q){
    // printf("Thread %d\n", q);
    for(int k = 2; k <= Kmax; ++k){
        F[q].emplace_back(new UnionFind(rows_to_consider[k - 2]));
    }
    for(int i = q; i < all_cliques.size(); i += NUMTHEADS){
        for(int j = i + 1; j < all_cliques.size(); ++j){
            int ov = Overlap(*all_cliques[i], *all_cliques[j]);
            for(int k = 2; k <= ov + 1; ++k){
                F[q][k - 2] -> Union(i, j);
            }
        }
    }
}

void CONNECT_ME(UnionFind *A, UnionFind *B){
    for(int i = 0; i < B -> fa.size(); ++i){
        if(B -> Find(i) != i){
            A -> Union(i, B -> fa[i]);
        }
    }
}

vector<UnionFind *> COSpoc(){

    tbb::parallel_for(0, NUMTHEADS, 1, [=](int i){ 
        // int q = tbb::task_arena::current_thread_index();
        COSpoc_work(i);
    });
    // #pragma omp parallel num_threads(NUMTHEADS) 
    // {
    //     int q = omp_get_thread_num();
    //     COSpoc_work(q, F[q]);
    // }
    // for(int q = 0; q < NUMTHEADS; ++q){
    //     COSpoc_work(q, F[q]);
    // }

    for(int k = 2; k <= Kmax; ++k){
        for(int q = 1; q < NUMTHEADS; ++q){
            CONNECT_ME(F[0][k - 2], F[q][k - 2]);
        }
    }

    for(int k = 2; k <= Kmax; ++k){
        for(int q = 1; q < NUMTHEADS; ++q){
            delete F[q][k - 2];
        }
    }
    return F[0];
}

vector<int> Buff;
long long base = 0;

inline int SlideID(int i, int j, int n){
    return (2 * n - i - 1) * i / 2 + (j - i) - base;
}

inline int SlideRead(int i, int j, int n){
    return Buff[SlideID(i, j, n)];
}

inline void SlideWrite(int v, int i, int j, int n){
    Buff[SlideID(i, j, n)] = v;
}

inline int SlideCalc(double s, double w, double n){
    double a = -1;
    double b = 2 * n - 1;
    double c = - (2 * n - s - 1) * s - 2 * w;
    double delta = b * b - 4 * a * c;
    if(delta < 0) return n;
    int ans1 = floor((-b + sqrt(delta)) / 2 * a);
    int ans2 = floor((-b - sqrt(delta)) / 2 * a);
    int ans = n;
    if(ans1 > s && ans1 < n) ans = min(ans, ans1);
    if(ans2 > s && ans2 < n) ans = min(ans, ans2);
    return ans;
}

void COS_calc(int q, int s, int e){
    // printf("Thread %d\n", q);
    int n = all_cliques.size();
    for(int i = s + q; i < e; i += NUMTHEADS){
        for(int j = i + 1; j < n; ++j){
            int ov = Overlap(*all_cliques[i], *all_cliques[j]);
            SlideWrite(ov, i, j, n);
        }
    }
}

using my_mutex_t = tbb::spin_mutex;

void COS_proc(int q, const vector<UnionFind *> &F_global, int s, int e){
    // printf("Thread %d\n", q);
    int n = all_cliques.size();
    UnionFind * F_q = new UnionFind(n);
    static vector<my_mutex_t> my_mutex(Kmax - 1);
    for(int k = Kmax; k >= 2; --k){
        for(int i = s + q; i < e; i += NUMTHEADS){
            for(int j = i + 1; j < rows_to_consider[k - 2]; ++j){
                int ov = SlideRead(i, j, n);
                if(ov == k - 1){
                    F_q -> Union(i, j);
                    // SlideWrite(0, i, j, n);
                }
            }
        }
        // #pragma omp critical
        {
            my_mutex_t::scoped_lock lock(my_mutex[k - 2]);
            // cerr << q << " begin " << k << endl;
            CONNECT_ME(F_global[k - 2], F_q);
            // cerr << q << " finish " << k << endl;
        }
    }
    delete F_q;
}

vector<UnionFind *> COS(int limit){
    vector<UnionFind *> F_global;
    for(int k = 2; k <= Kmax; ++k){
        F_global.emplace_back(new UnionFind(rows_to_consider[k - 2]));
    }
    int s = 0;
    int e = 0;
    int n = all_cliques.size();
    long long tot = (long long)(n - 1) * n / 2;
    if(limit < n) limit = n;
    if(limit > tot) limit = tot;
    Buff.resize(limit);
    while(s < n - 1){
        e = SlideCalc(s, limit, n);
        // cerr << s << " " << e << endl;
        tbb::parallel_for(0, NUMTHEADS, 1, [ = ](int i){ 
            // int q = tbb::task_arena::current_thread_index();
            COS_calc(i, s, e);
        });
        // #pragma omp parallel num_threads(NUMTHEADS) 
        // {
        //     int q = omp_get_thread_num();
        //     COS_calc(q, s, e);
        // }
        // cerr << "ok1" << endl;

        tbb::parallel_for(0, NUMTHEADS, 1, [ = ](int i){ 
            // int q = tbb::task_arena::current_thread_index();
            COS_proc(i ,F_global, s, e);
        });
        // #pragma omp parallel num_threads(NUMTHEADS) 
        // {
        //     int q = omp_get_thread_num();
        //     COS_proc(q, F_global, s, e);
        // }
        // cerr << "ok2" << endl;
        s = e;
        base = (long long)(2 * n - s - 1) * s / 2;
        // cerr << 100.0 * base / tot  << "%" << endl;
    }
    return F_global;
}

void Extract(const vector<UnionFind *> &communities, Graph *G){
    for(int k = 2; k <= Kmax; ++k){
        unordered_map<int, vector<int>> tmp;
        for(int i = 0; i < communities[k - 2] -> fa.size(); ++i){
            tmp[communities[k - 2] -> Find(i)].push_back(i);
        }
        printf("%d-clique-communities: %d\n", k, tmp.size());
        // vector<vector<int> > k_clique_communities;
        // for(auto &node: tmp){
        //     set<int> S;
        //     for(auto &id: node.second){
        //         for(auto &x: *all_cliques[id]){
        //             S.insert(x);
        //         }
        //     }
        //     vector<int>community;
        //     for(auto &x: S){
        //         community.push_back(G -> ID[x]);
        //         // printf("%d ", G -> ID[x]);
        //     }
        //     k_clique_communities.emplace_back(community);
        //     // puts("");
        // }
        // sort(k_clique_communities.begin(), k_clique_communities.end());
        // for(auto &community : k_clique_communities){
        //     for(auto &x: community){
        //         printf("%d ", x);
        //     }
        //     puts("");
        // }
        delete communities[k - 2];
    }
}

int main(){
    // freopen(".out","w",stdout);
    Graph G; G.init();
    FindMaximalCliques(&G);
    auto st = tbb::tick_count::now();
    vector<UnionFind *> Communities = COSpoc();
    // vector<UnionFind *> Communities = COS(1e8);
    auto ed = tbb::tick_count::now();

    cerr << (ed - st).seconds() << endl;
    Extract(Communities, &G);
    return 0;
}
