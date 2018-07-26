// Comparators for tests. Avoid templates for test compilation speedup.

#include <cmath>
#include <string>
#include <algorithm>
#include <memory>
#include <vector>

#define DEBUG

#ifdef DEBUG
#include <iostream>
#endif

using namespace std;

bool same_output(double a, double b) {
    return a == b || fabs(a - b) < 1e-6 || (a != 0 && b != 0 && fabs((a - b) / min(fabs(a), fabs(b))) < 1e-6);
}

string string_trim_all(string s) {
    string res = s;
    res.erase(res.begin(), std::find_if(res.begin(), res.end(), [](int ch) {
        return !std::isspace(ch);
    }));
    res.erase(find_if(res.rbegin(), res.rend(), [](int ch) {
        return !isspace(ch);
    }).base(), res.end());
    return res;
}

bool same_output(string a, string b) {
    return string_trim_all(a) == string_trim_all(b);
}

#define COMPARE_VECTORS(a, b) { \
    if (a->size() != b->size()) \
        return false; \
    for (size_t i = 0; i < a->size(); ++i) \
        if (!same_output(a->at(i), b->at(i))) \
            return false; \
    return true; \
}

bool same_output(shared_ptr<vector<long> > a, shared_ptr<vector<long> > b) COMPARE_VECTORS(a, b)

bool same_output(shared_ptr<vector<double> > a, shared_ptr<vector<double> > b) COMPARE_VECTORS(a, b)

bool same_output(shared_ptr<vector<string> > a, shared_ptr<vector<string> > b) COMPARE_VECTORS(a, b)

bool same_output(shared_ptr<vector<shared_ptr<vector<long> > > > a, shared_ptr<vector<shared_ptr<vector<long> > > >  b) COMPARE_VECTORS(a, b)

bool same_output(shared_ptr<vector<shared_ptr<vector<double> > > > a, shared_ptr<vector<shared_ptr<vector<double> > > >  b) COMPARE_VECTORS(a, b)

bool same_output(shared_ptr<vector<shared_ptr<vector<string> > > > a, shared_ptr<vector<shared_ptr<vector<string> > > >  b) COMPARE_VECTORS(a, b)


#ifdef DEBUG

#define PRINT_VECTOR { \
    cout << "["; \
    for (auto el : *a) { \
        print_output(el); \
        cout << ", "; \
    } \
    cout << "]"; \
}

void print_output(string a) {cout << '"' << a << '"';}

void print_output(double a) {cout << a;}

void print_output(shared_ptr<vector<long> > a) PRINT_VECTOR
void print_output(shared_ptr<vector<double> > a) PRINT_VECTOR
void print_output(shared_ptr<vector<string> > a) PRINT_VECTOR
void print_output(shared_ptr<vector<shared_ptr<vector<long> > > >a) PRINT_VECTOR
void print_output(shared_ptr<vector<shared_ptr<vector<double> > > >a) PRINT_VECTOR
void print_output(shared_ptr<vector<shared_ptr<vector<string> > > >a) PRINT_VECTOR

#endif
