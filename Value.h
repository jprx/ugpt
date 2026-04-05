#pragma once
#include <vector>
#include <iostream>
#include <set>

class Value {
public:
  // d: data at this node
  // grad: derivative of loss wrt this node
  // dl/dr: derivative wrt left / right inputs
  // l/r: left / right inputs
  double d, grad;
  double dl, dr;
  Value *l, *r;

  Value(float d_in, Value *l_in = NULL, Value *r_in = NULL, float dl_in = 0.0, float dr_in = 0.0) :
    d(d_in), l(l_in), r(r_in), dl(dl_in), dr(dr_in), grad(0.0) {}

  friend std::ostream& operator<<(std::ostream &out, const Value &v) {
    return out << v.d;
  }

  void build_topo(Value *v, std::set<Value*> &visited, std::vector<Value*> &topo) {
    if (!visited.contains(v)) {
      visited.insert(v);
      if (v->l) build_topo(v->l, visited, topo);
      if (v->r) build_topo(v->r, visited, topo);
      topo.push_back(v);
    }
  }

  void backprop() {
    std::set<Value*> visited;
    std::vector<Value*> topo;

    build_topo(this, visited, topo);
    this->grad = 1;
    for (std::vector<Value*>::reverse_iterator riter = topo.rbegin(); riter != topo.rend(); ++riter) {
      if ((*riter)->l) (*riter)->l->grad += (*riter)->dl * (*riter)->grad;
      if ((*riter)->r) (*riter)->r->grad += (*riter)->dr * (*riter)->grad;
    }
  }
};

static inline Value *operator+(Value &a, Value &b) { return new Value(a.d + b.d, &a, &b, 1, 1); }
static inline Value *operator-(Value &a, Value &b) { return new Value(a.d - b.d, &a, &b, 1,-1); }
static inline Value *operator*(Value &a, Value &b) { return new Value(a.d * b.d, &a, &b, b.d, a.d); }
static inline Value *operator+(Value &a, double b) { Value *v = new Value(b); return a + (*v); }
static inline Value *operator-(Value &a, double b) { Value *v = new Value(b); return a - (*v); }
static inline Value *operator*(Value &a, double b) { Value *v = new Value(b); return a * (*v); }
static inline Value *operator^(Value &a, double b) { return new Value(pow(a.d, b), &a, NULL, b*pow(a.d, b-1)); }
static inline Value *operator-(Value &a) { return a * (-1.0); }
static inline Value *operator/(Value &a, Value &b) { return a * *(b^(-1)); }
static inline Value *operator/(Value &a, double b) { Value *v = new Value(b); return a / (*v); }
static inline Value *exp(Value &a) { return  new Value(exp(a.d), &a, NULL, exp(a.d)); }
static inline Value *log(Value &a) { return  new Value(log(a.d), &a, NULL, 1/a.d); }
static inline Value *relu(Value &a) { return new Value(a.d > 0 ? a.d : 0, &a, NULL, a.d > 0 ? 1 : 0); }
