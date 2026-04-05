#include <vector>
#include <assert.h>
#include <string>
#include <fstream>
#include <algorithm>
#include <random>
#include <span>
#include <map>
#include "Value.h"

size_t bos = 26;
size_t vocab_size = 27;

size_t num_steps = 1000;
size_t n_embd = 16;
size_t n_head = 4;
size_t n_layer = 1;
size_t block_size = 16;
size_t head_dim = n_embd / n_head;

using std::string;
using std::map;
using std::vector;

typedef vector<vector<Value*>> matrix;

void log_vector(std::span<Value*> x) {
  for (size_t i = 0; i < x.size(); i++) {
    std::cout << *x[i] << "\n";
  }
}

vector<string> get_data(string path) {
  vector<string> dat;
  std::ifstream f(path);
  if (!f.is_open()) return dat;

  string line;
  while (getline(f, line)) {
    dat.push_back(line);
  }

  std::shuffle(dat.begin(), dat.end(), std::default_random_engine(std::random_device()()));

  return dat;
}

// Random matrix that takes n_in dim vector -> n_out dim vector
matrix randmat(size_t n_out, size_t n_in) {
  vector<vector<Value*>> m (n_out, vector<Value*> (n_in, 0));

  std::normal_distribution<double> d(0, 0.08);
  std::random_device rd;

  for (size_t i = 0; i < n_out; i++) {
    for (size_t j = 0; j < n_in; j++) {
      m[i][j] = new Value(d(rd));
    }
  }

  return m;
}

map<string, matrix> build_parameters() {
  map<string, matrix> state_dict;

  state_dict["wte"] = randmat(vocab_size, n_embd);
  state_dict["wpe"] = randmat(block_size, n_embd);
  state_dict["lm_head"] = randmat(vocab_size, n_embd);

  state_dict["layer0.attn_wq"] = randmat(n_embd, n_embd);
  state_dict["layer0.attn_wk"] = randmat(n_embd, n_embd);
  state_dict["layer0.attn_wv"] = randmat(n_embd, n_embd);
  state_dict["layer0.attn_wo"] = randmat(n_embd, n_embd);
  state_dict["layer0.mlp_fc1"] = randmat(4 * n_embd, n_embd);
  state_dict["layer0.mlp_fc2"] = randmat(n_embd, 4 * n_embd);
  return state_dict;
}

vector<Value*> matmul(matrix &w, vector<Value*> &x) {
  vector<Value*> v(w.size());
  for (size_t i = 0; i < w.size(); i++) {
    vector<Value*> wo = w[i];

    assert(wo.size() == x.size());

    v[i] = new Value(0);
    for (size_t j = 0; j < wo.size(); j++) {
      v[i] = *v[i] + *(*x[j] * *wo[j]);
    }
  }

  return v;
}

vector<Value*> softmax(vector<Value*> &x) {
  double max_val = (*std::max_element(x.begin(), x.end(), [](Value *v1, Value *v2) {return v1->d < v2->d;}))->d;

  vector<Value*> exps = vector<Value*>(x.size());

  for (size_t i = 0; i < x.size(); i++) {
    exps[i] = exp(*(*x[i] - max_val));
  }

  Value *exp_sum = new Value(0);
  for (size_t i = 0; i < exps.size(); i++) {
    exp_sum = *exp_sum + *exps[i];
  }

  for (size_t i = 0; i < exps.size(); i++) {
    exps[i] = *exps[i] / *exp_sum;
  }

  return exps;
}

vector<Value*> rmsnorm(vector<Value*> &x) {
  vector<Value*> o(x.size());

  Value *ms = new Value(0);
  for (size_t i = 0; i < x.size(); i++) {
    ms = *ms + *(*x[i] * *x[i]);
  }

  ms = *ms / x.size();
  Value *scale = *(*ms + 1e-5) ^ -0.5;

  for (size_t i = 0; i < x.size(); i++) {
    o[i] = *x[i] * *scale;
  }

  return o;
}

vector<Value*> gpt(
  size_t token_id,
  size_t pos_id,
  vector<vector<Value*>> &keys,
  vector<vector<Value*>> &values,
  map<string, matrix> &state
) {
  vector<Value*> &tok_emb = state["wte"][token_id];
  vector<Value*> &pos_emb = state["wpe"][pos_id];
  vector<Value*> x(tok_emb.size());

  for (size_t i = 0; i < x.size(); i++) {
    x[i] = *tok_emb[i] + *pos_emb[i];
  }
  x = rmsnorm(x);

  // 1. Multi-Head Attention Block
  vector<Value*> x_residual = x;
  x = rmsnorm(x);

  vector<Value*> q = matmul(state["layer0.attn_wq"], x);
  vector<Value*> k = matmul(state["layer0.attn_wk"], x);
  vector<Value*> v = matmul(state["layer0.attn_wv"], x);
  keys.push_back(k);
  values.push_back(v);
  vector<Value*> x_attn;

  for (size_t h = 0; h < n_head; h++) {
    size_t hs = h * head_dim;
    std::span<Value*> q_h = std::span(q.begin()+hs, q.begin()+hs+head_dim);

    std::vector<std::span<Value*>> k_h, v_h;
    for (size_t ki = 0; ki < keys.size(); ki++) {
      k_h.push_back(std::span(keys[ki].begin()+hs, keys[ki].begin()+hs+head_dim));
    }
    for (size_t vi = 0; vi < keys.size(); vi++) {
      v_h.push_back(std::span(values[vi].begin()+hs, values[vi].begin()+hs+head_dim));
    }

    vector<Value*> attention_logits(k_h.size());
    for (size_t t = 0; t < k_h.size(); t++) {
      attention_logits[t] = new Value(0);
      for (size_t j = 0; j < head_dim; j++) {
        attention_logits[t] = *attention_logits[t] + *(*q_h[j] * *k_h[t][j]);
      }
      attention_logits[t] = *attention_logits[t] / sqrt((double)head_dim);
    }

    vector<Value*> attention_weights = softmax(attention_logits);

    vector<Value*> head_out(head_dim);
    for (size_t j = 0; j < head_dim; j++) {
      head_out[j] = new Value(0);
      for (size_t t = 0; t < v_h.size(); t++) {
        head_out[j] = *head_out[j] + *(*attention_weights[t] * *v_h[t][j]);
      }
    }

    for (Value *v : head_out) {
      x_attn.push_back(v);
    }
  }

  x = matmul(state["layer0.attn_wo"], x_attn);

  for (size_t i = 0; i < x.size(); i++) {
    x[i] = *x[i] + *x_residual[i];
  }

  // 2. MLP
  x_residual = x;
  x = rmsnorm(x);
  x = matmul(state["layer0.mlp_fc1"], x);

  for (size_t i = 0; i < x.size(); i++) {
    x[i] = relu(*x[i]);
  }

  x = matmul(state["layer0.mlp_fc2"], x);

  for (size_t i = 0; i < x.size(); i++) {
    x[i] = *x[i] + *x_residual[i];
  }

  vector<Value*> logits = matmul(state["lm_head"], x);
  return logits;
}

vector<Value*> get_params_from_state(map<string, matrix> &state) {
  vector<Value*> params;
  for (const auto &[k,v] : state) {
    for (vector<Value*> col : v) {
      for (Value *val : col) {
        params.push_back(val);
      }
    }
  }

  return params;
}

vector<int> tokenize(string s) {
  vector<int> o;
  o.push_back(bos);
  for (char c : s) {
    o.push_back((int)c-0x60);
  }
  o.push_back(bos);
  return o;
}

int main() {
  vector<string> data = get_data("names.txt");
  map<string, matrix> state = build_parameters();
  vector<Value*> params = get_params_from_state(state);

  // 1. Training loop
  double learning_rate = 0.01;
  double beta1 = 0.85;
  double beta2 = 0.99;
  double eps_adam = 1e-8;

  vector<double> m(params.size(), 0);
  vector<double> v(params.size(), 0);

  for (size_t step = 0; step < num_steps; step++) {
    string doc = data[step];

    vector<int> tokens = tokenize(doc);

    size_t n = std::min(block_size, tokens.size() - 1);

    vector<vector<Value*>> keys, values;
    vector<Value*> losses;

    for (size_t pos_id = 0; pos_id < n; pos_id++) {
      size_t token_id = tokens[pos_id];
      size_t target_id = tokens[pos_id+1];

      vector<Value*> logits = gpt(token_id, pos_id, keys, values, state);
      vector<Value*> probs = softmax(logits);

      Value *loss_t = -(*log(*probs[target_id]));
      losses.push_back(loss_t);
    }

    Value *loss = new Value(0);
    for (size_t i = 0; i < losses.size(); i++) {
      loss = *loss + *losses[i];
    }
    loss = *loss / n;

    loss->backprop();

    double lr_t = learning_rate * (1 - (double)step / (double)num_steps);
    for (size_t i = 0; i < params.size(); i++) {
      m[i] = beta1 * m[i] + (1 - beta1) * params[i]->grad;
      v[i] = beta2 * v[i] + (1 - beta2) * pow(params[i]->grad, 2);

      double m_hat = m[i] / (1 - pow(beta1, step + 1));
      double v_hat = v[i] / (1 - pow(beta2, step + 1));

      params[i]->d -= lr_t * m_hat / (sqrt(v_hat) + eps_adam);

      params[i]->grad = 0;
    }

    std::cout << "Loss: " << *loss;
    std::cout << "\t" << step << " (" << doc << ")\n";
  }

  // 2. Inference
  double temperature = 0.5;
  std::cout << " --- Inference ---\n";
  for (size_t sample_idx = 0; sample_idx < 20; sample_idx++) {
    vector<vector<Value*>> keys, values;
    int token_id = bos;
    string sample;

    for (size_t pos_id = 0; pos_id < block_size; pos_id++) {
      vector<Value*> logits = gpt(token_id, pos_id, keys, values, state);

      for (size_t i = 0; i < logits.size(); i++) {
        logits[i] = *logits[i] / temperature;
      }

      vector<Value*> probs = softmax(logits);

      vector<double> probs_double(probs.size());
      for (size_t i = 0; i < probs.size(); i++) {
        probs_double[i] = probs[i]->d;
      }

      std::discrete_distribution<> d(probs_double.begin(), probs_double.end());
      std::random_device dev;
      token_id = d(dev);

      if (bos == token_id) break;

      sample.push_back((char)(token_id+0x60));
    }

    std::cout << sample << "\n";
  }

  return 0;
}
