#include <string>
#include <algorithm>
#include <memory>

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
