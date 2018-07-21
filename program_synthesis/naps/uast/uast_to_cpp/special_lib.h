#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <memory>
//#include <initializer_list>
#include <utility>
#include <type_traits>
#include <stdexcept>

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

// String operations.
template<typename T>
int string_find(string str, T sub) {
    const size_t pos = str.find(sub);
    return (pos == string::npos)?-1:pos;
}

template<typename T>
int string_find_last(string str, T sub) {
    const size_t pos = str.rfind(sub);
    return (pos == string::npos)?-1:pos;
}

template<typename T>
string to_str(T x) {
    if (is_convertible<O, string>::value)
        old_sub = static_cast<string>(old_sub_);
    else if (is_convertible<O, char>::value)
        old_sub = string(1, static_cast<char>(old_sub_));
    else throw invalid_argument("Expected string.")
}

template<typename O, typename N>
string string_replace_one(string str, O old_sub_, N new_sub_) {
    string old_sub = to_str(old_sub_);
    string new_sub = to_str(new_sub_);
    auto pos = str.find(old_sub)
    if (pos == string::npos) return str;
    string res = str;
    return res.replace(res.begin()+pos, res.begin()+pos+old_sub.length(),
        new_sub.begin(), new_sub.end());
}

template<typename O, typename N>
string string_replace_all(string str, O old_sub_, N new_sub_) {
    string old_sub = to_str(old_sub_);
    string new_sub = to_str(new_sub_);

    size_t pos = -1;
    string res = str;
    while((pos = res.find(old_sub, pos+1)) != string::npos) {
        res = res.replace(res.begin()+pos, res.begin()+pos+old_sub.length(),
        new_sub.begin(), new_sub.end());
    }
    return res;
}

template<typename A, typename B>
string concat(A a, B b) {
    return to_str(a) + to_str(b);
}

template<typename T>
shared_ptr<vector<T> > array_concat(shared_ptr<vector<T> > a, shared_ptr<vector<T> > b) {
    shared_ptr<vector<T> > res =
}