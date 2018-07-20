#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <memory>
//#include <initializer_list>
#include <utility>

using namespace std;

template<typename T>
shared_ptr<vector<T> > reverse(shared_ptr<vector<T> > c) {
  shared_ptr<vector<T> > res = make_shared<vector<T> >(c->begin(), c->end());
  reverse(res->begin(), res->end());
  return res;
}

string reverse(string c) {
  string res = c;
  reverse(c.begin(), c.end());
  return res;
}

// Special comparators for containers that make them behave like those in Python. Allows comparing vectors, sets, maps,
// vectors of vectors, vectors of sets of maps, etc.
template<typename T>
bool special_comparator(T a, T b) { return a < b; }

template<typename T>
bool special_comparator(const shared_ptr<vector<T> >& a, const shared_ptr<vector<T> >& b) {
    return lexicographical_compare(a->begin(), a->end(), b->begin(), b->end(),
    static_cast<bool(*)(T, T)>(&special_comparator<T>));
}

template<typename K, typename V>
bool special_comparator(const shared_ptr<map<K, V> >& a, const shared_ptr<map<K, V> >& b) {
  return lexicographical_compare(a->begin(), a->end(), b->begin(), b->end(),
    static_cast<bool(*)(pair<K, V>, pair<K, V>)>(&special_comparator<pair<K, V> >));
}

template<typename T>
bool special_comparator(const shared_ptr<set<T> >& a, const shared_ptr<set<T> >& b) {
  return lexicographical_compare(a->begin(), a->end(), b->begin(), b->end(),
    static_cast<bool(*)(T, T)>(&special_comparator<T>));
}


template<typename T>
shared_ptr<vector<T> > sort(shared_ptr<vector<T> > c) {
    shared_ptr<vector<T> > res = make_shared<vector<T> >(c->begin(), c->end());
    stable_sort(res->begin(), res->end(), static_cast<bool(*)(T, T)>(&special_comparator<T>));
    return res;
}

template<typename K, typename V>
shared_ptr<vector<tuple<K, V> > > sort(shared_ptr<map<K, V> > c) {
    shared_ptr<vector<tuple<K, V> > > res = make_shared<vector<tuple<K, V> > >();
    for (auto el: *c)
        res->push_back(make_tuple(el.first, el.second));
    stable_sort(res->begin(), res->end(),
    static_cast<bool(*)(tuple<K, V>, tuple<K, V>)>(&special_comparator<tuple<K, V> >));
    return res;
}


template<typename T, typename Compare>
shared_ptr<vector<T> > sort_cmp(shared_ptr<vector<T> > c, Compare comp) {
    shared_ptr<vector<T> > res = make_shared<vector<T> >(c->begin(), c->end());
    stable_sort(res->begin(), res->end(), &comp);
    return res;
}

template<typename T>
void fill(shared_ptr<vector<T> > c, const T& val) {
    fill(c->begin(), c->end(), val);
}

template<typename T>
shared_ptr<vector<T> > copy_range(shared_ptr<vector<T> > c, int from, int to) {
    if (from >= to)
        return make_shared<vector<T> >();
    if (to < 0) to = c.size() + to;
    to = min(to, c.size());
    from = max(0, from);
    return make_shared<vector<T> >(c->begin()+from, c->begin()+to);
}