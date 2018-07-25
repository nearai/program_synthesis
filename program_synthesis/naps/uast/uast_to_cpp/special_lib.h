#ifndef _SPECIAL_LIB_H_
#define _SPECIAL_LIB_H_

#include <cctype>
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <memory>
#include <regex>
#include <cmath>
#include <locale>
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

// More flexible min, max, pow operations.
// Note, current C++ min/max ops have some deficiencies that make them more limited than Python
// variants, see:
// https://www.reddit.com/r/cpp/comments/5p81eq/why_dont_stdminmaxclamp_use_common_type_to_work/
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2199.html
template<typename T1, typename T2>
typename std::common_type<T1,T2>::type
min_(T1 t1, T2 t2){ return (t1 < t2) ? t1 : t2; }

template<typename T1, typename T2>
typename std::common_type<T1,T2>::type
max_(T1 t1, T2 t2){ return (t1 > t2) ? t1 : t2; }

template<typename T1, typename T2>
typename std::common_type<T1,T2>::type
pow_(T1 t1, T2 t2){ return static_cast<typename std::common_type<T1,T2>::type >(pow(t1, t2)); }

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

string to_string(string str) {
    return str;
}

string sort(string str) {
    string res = str;
    stable_sort(res.begin(), res.end());
    return res;
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

template<typename T, typename E>
void fill(shared_ptr<vector<T> > c, const E& val) {
    fill(c->begin(), c->end(), static_cast<T>(val));
}

template<typename T>
shared_ptr<vector<T> > copy_range(shared_ptr<vector<T> > c, int from, int to) {
    if (from >= to)
        return make_shared<vector<T> >();
    if (to < 0) to = c->size() + to;
    to = min(to, c->size());
    from = max(0, from);
    return make_shared<vector<T> >(c->begin()+from, c->begin()+to);
}

// String operations.
string tolower(string str) {
    string res = str;
    for (size_t i = 0; i< str.length(); ++i) {
        res[i] = tolower(res[i]);
    }
    return res;
}

string toupper(string str) {
    string res = str;
    for (size_t i = 0; i< str.length(); ++i) {
        res[i] = toupper(res[i]);
    }
    return res;
}

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

// SFINAE, see https://stackoverflow.com/questions/25284499/how-does-stdenable-if-work
template<typename T>
typename enable_if<is_convertible<T, string>::value, string>::type
to_str(T x) {
    return static_cast<string>(x);
}

template<typename T>
typename enable_if<is_convertible<T, char>::value, string>::type
to_str(T x) {
    return string(1, static_cast<char>(x));
}

template<typename O, typename N>
string string_replace_one(string str, O old_sub_, N new_sub_) {
    string old_sub = to_str(old_sub_);
    string new_sub = to_str(new_sub_);
    auto pos = str.find(old_sub);
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
    shared_ptr<vector<T> > res = make_shared<vector<T> >(a->begin(), a->end());
    res->insert(res->end(), b->begin(), b->end());
    return res;
}

template<typename T>
string string_insert(string str, int pos, T sub_) {
    string sub = to_str(sub_);
    string res = str;
    res = res.insert(pos, sub);
    return res;
}

shared_ptr<vector<string> > string_split(string str, string delimiters) {
    if (delimiters == "") {
        // Note, this code might need more work to mimic the behavior of the equivalent Python function.
        shared_ptr<vector<string> > res = make_shared<vector<string> >();
        for (auto c : str)
            res->push_back(to_str(c));
        return res;
    }
    // Following https://www.quora.com/How-do-I-split-a-string-by-space-into-an-array-in-c++
    string regex_delimiters;
    for (size_t i = 0; i < delimiters.size(); ++i) {
        const char c = delimiters[i];
        if (c == '|' || c == '\\' || c == '+' || c == '(' || c == ')' || c == ',' || c == '[' || c == ']') {
            regex_delimiters += "\\" + to_str(c);
        } else {
            regex_delimiters += to_str(c);
        }
        if (i != delimiters.size()-1)
            regex_delimiters += "|";
    }
    regex re(regex_delimiters);
    shared_ptr<vector<string> > res = make_shared<vector<string> >(
        sregex_token_iterator(str.begin(), str.end(), re, -1),
        sregex_token_iterator());

    // Remove empty strings.
    res->erase(remove(res->begin(), res->end(), ""), res->end());
    return res;
}

string string_trim(string s) {
    string res = s;
    res.erase(find_if(res.rbegin(), res.rend(), [](int ch) {
        return !isspace(ch);
    }).base(), res.end());
    return res;
}

string substring(string s, long from, long to) {
    // Fix indices to match the behavior of the Python function.
    from = max(0L, from);
    if (to < 0) to = s.size() + to;
    to = min(static_cast<long>(s.size()), to);
    to = max(from, to);
    long len = to - from;
    return s.substr(from, len);
}

string copy_range(string s, int from, int to) {
    return substring(s, from, to);
}

string substring_end(string s, long from) {
    return substring(s, from, s.size());
}

template<typename T> struct is_shared_ptr : std::false_type {};
template<typename T> struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

template<typename T, typename E>
typename enable_if<is_shared_ptr<E>::value, shared_ptr<vector<T> > >::type
array_push(shared_ptr<vector<T> > v, E a) {
    v->push_back(move(a));
    return v;
}

template<typename T, typename E>
typename enable_if<!is_shared_ptr<E>::value, shared_ptr<vector<T> > >::type
array_push(shared_ptr<vector<T> > v, E a) {
    v->push_back(a);
    return v;
}

template<typename T>
T array_pop(shared_ptr<vector<T> > v) {
    T res = v->back();
    v->pop_back();
    return res;
}

template<typename T, typename E>
shared_ptr<vector<T> > array_insert(shared_ptr<vector<T> > v, long pos, E a) {
    v->insert(v->begin()+pos, a);
    return v;
}

template<typename T>
T array_remove_idx(shared_ptr<vector<T> > v, long pos) {
    T res = v->at(pos);
    v->erase(v->begin()+pos);
    return res;
}

template<typename T, typename E>
T array_remove_value(shared_ptr<vector<T> > v, E value_) {
    T value = static_cast<T>(value_);
    T res = value;
    for (size_t pos = 0; pos < v->size(); ++pos) {
        auto& el = v->at(pos);
        if (!special_comparator(el, value) && !special_comparator(value, el)) {
            v->erase(v->begin()+pos);
            return res;
        }
    }
    return res;
}

template<typename T, typename E>
int array_find(shared_ptr<vector<T> > v, E value_) {
    T value = static_cast<T>(value_);
    for (size_t pos = 0; pos < v->size(); ++pos) {
        auto& el = v->at(pos);
        if (!special_comparator(el, value) && !special_comparator(value, el))
            return pos;
    }
    return -1;
}

template<typename T, typename E>
int array_find(shared_ptr<vector<T> > v, E value_, int start_pos) {
    T value = static_cast<T>(value_);
    for (size_t pos = start_pos; pos < v->size(); ++pos) {
        auto& el = v->at(pos);
        if (!special_comparator(el, value) && !special_comparator(value, el))
            return pos;
    }
    return -1;
}

template<typename T, typename E>
bool contains(shared_ptr<vector<T> > v, E value) {
    return array_find(v, value) != 1;
}

template<typename K, typename V, typename E>
bool contains(const shared_ptr<map<K, V> >& v, E value) {
    return v->find(value) != v->end();
}

template<typename T, typename E>
bool contains(const shared_ptr<set<T> >& v, E value) {
    return v->find(value) != v->end();
}

template<typename K, typename V>
shared_ptr<vector<K> > map_keys(const shared_ptr<map<K, V> >& v) {
    shared_ptr<vector<K> > res = make_shared<vector<K> >();
    for (auto& el : *v) res->push_back(el.first);
    return res;
}

template<typename K, typename V>
shared_ptr<vector<V> > map_values(const shared_ptr<map<K, V> >& v) {
    shared_ptr<vector<V> > res = make_shared<vector<V> >();
    for (auto& el : *v) res->push_back(el.second);
    return res;
}

template<typename T>
shared_ptr<vector<T> > array_initializer(initializer_list<T> elements) {
    return make_shared<vector<T> >(elements);
}


// SFINAE with variadic templates for array initialization.
template<typename T, typename D>
typename enable_if<is_same<T, vector<typename T::value_type,
                                     typename T::allocator_type > >::value,
                                     shared_ptr<T> >::type
init_darray(D dim) {
    return make_shared<T>(dim);
}

template<typename T, typename D, typename... Dims>
typename enable_if<is_same<T, vector<typename T::value_type,
                                     typename T::allocator_type > >::value,
                                     shared_ptr<T> >::type
init_darray(D dim, Dims... dims) {
    auto res = make_shared<T>(dim);
    for (size_t i = 0; i<res->size(); ++i)
        res->at(i) = init_darray<typename T::value_type::element_type>(dims...);
    return res;
}


template<typename T>
typename enable_if<is_shared_ptr<T>::value, long >::type
size(T c) {
    return c->size();
}

template<typename T>
typename enable_if<!is_shared_ptr<T>::value, long >::type
size(T c) {
    return c.size();
}

template<typename T, typename O>
shared_ptr<T> from_iterable(O orig) {
    return make_shared<T>(orig->begin(), orig->end());
}
#endif
