---
layout: post
title: How to convert PyTorch Models to GGML for Backend Agnostic Local Inference
date: 2024-07-10 1:17:20 -0300
description: Guide for converting PyTorch models to GGML # Add post description (optional)
img: matterhorn.png # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [ggml, PyTorch, AI, FOSS]
---


## Introduction

GGML is a C++ tensor library that supports a variety of backends; largely known for the dependent conversions llama.cpp and whisper.cpp. I recently converted the text to speech library tortoise-tts to GGML, so I have acquired some familiarity with converting arbitrary PyTorch code to GGML. In this blog post, I hope to share some of the more general techniques I used, to make the task of converting a model to GGML seem less daunting.

This tutorial is tested with Ubuntu 22.04, though in principle it should be possible on any platform, though build tools may differ. 


## Basic Inference in GGML 

So let's start simple, It's best to first get something running in GGML before increasing complexity. First, clone ggml, then run the following commands to compile for CPU:

````bash
mkdir build
cd build
cmake ..
make
````

I'd recommend starting with simple-backend.cpp in the examples folder in ggml. Make sure you can compile it, then run it from `build` using:


````
./bin/simple-backend
````

Initially, the output should look like:

````
main: compute buffer size: 0.0938 KB
mul mat (4 x 3) (transposed result):
[ 60.00 110.00 54.00 29.00
 55.00 90.00 126.00 28.00
 50.00 54.00 42.00 64.00 ]

````


## A Simple PyTorch Module

For the sake of this tutorial, I have created a simple PyTorch module. This process should work for any PyTorch module, so you can substitute your own.  You can copy the below code into a file called `simple.py`.


````python
# program to create a pytorch module that multiplies an input vector by a matrix then adds a bias and a ReLU activation
# The module is defined then run in the main function

import torch
import torch.nn as nn

class SimpleModule(nn.Module):
    def __init__(self, weight, bias):
        super(SimpleModule, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(weight))
        self.bias = nn.Parameter(torch.Tensor(bias))

    def forward(self, x):
        return torch.relu(torch.matmul(x, self.weight) + self.bias)

if __name__ == '__main__':
    weight = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    bias = [7, 8, 9]
    module = SimpleModule(weight, bias)

    torch.save( module, "./model.pt")

    x = torch.Tensor([1,1,1])
    y = module(x)
    print(y)
````

You can set up and activate a local virtualenv for the required dependencies with the following commands:

````bash
python3 -m venv env
source env/bin/activate
python3 -m pip install torch numpy
````

Then you can run `simple.py` as follows:

````
python3 ./simple.py
````

The output should look like:

````
tensor([19., 23., 27.], grad_fn=<ReluBackward0>)
````

This module takes an input vector, multiplies it by a weight matrix, then adds a bias and applies the ReLU activation. For our example, we will convert this module to GGML. We conveniently already saved the module in `model.pt` when we called `simple.py` so we can now consider how to convert this saved model to GGML. 


## Exporting a model to the GGML format

The following script will let us export our module to GGML, and with minor modifications this script should work for any PyTorch model saved in the pt format.

````python

import io
import os
import sys
import struct
import json
import code
import torch
import torch.nn as nn
import numpy as np
import base64
from pathlib import Path


class SimpleModule(nn.Module):
    def __init__(self, weight, bias):
        super(SimpleModule, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(weight))
        self.bias = nn.Parameter(torch.Tensor(bias))

    def forward(self, x):
        return torch.relu(torch.matmul(x, self.weight) + self.bias)



if len(sys.argv) < 2:
    print("Usage: convert-pt-to-ggml.py model.pt ]\n")
    sys.exit(1)

fname_inp   = Path(sys.argv[1])
dir_out     = "./"

# try to load PyTorch binary data
try:
    model_bytes = open(fname_inp, "rb").read()
    with io.BytesIO(model_bytes) as fp:
        checkpoint = torch.load(fp, map_location="cpu")
        #print(checkpoint.state_dict)
        #print(dir(checkpoint))
except Exception as e:
    print("Error: failed to load PyTorch model file:" , fname_inp)
    sys.exit(1)


list_vars = checkpoint.state_dict()


fname_out = dir_out + "simple-ggml-model.bin"


fout = open(fname_out,"wb")



fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex

#optionally you can write hyperparameters to file here,
""" 
example_hyperparam = 1
fout.write(struct.pack("i", example_hyperparam))
"""

for name in list(list_vars.keys()):

    data = list_vars[name].squeeze().numpy()
    print("Processing variable: " , name ,  " with shape: ", data.shape, " with type: ", data.dtype)

    n_dims = len(data.shape)

    data = data.astype(np.float32)
    ftype = 0 # set this to 1 for tensors that are float16

    str_ = name.encode('utf-8')
    print(ftype)
    fout.write(struct.pack("iii", n_dims, len(str_), ftype))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(str_)

    # data
    data.tofile(fout)


fout.close()

print("Done. Output file: " , fname_out)
print("")
````

First, save this script to a file called `convert.py`, then run it from the command line with the following invocation:

````
python3 ./convert.py model.pt
````

The output should look like:

````
Processing variable:  weight  with shape:  (3, 3)  with type:  float32
0
Processing variable:  bias  with shape:  (3,)  with type:  float32
0
Done. Output file:  ./simple-ggml-model.bin
````

Now you have your ggml version of the PyTorch model saved to disk. There might be cases where you want to also save float16 tensors to the ggml format, for these cases, you need to set
the data type to `np.float16` and set `ftype` to 1. Another trick that comes in handy when converting large models is creating ggml files that only contain some of the tensors, so you can test some of your logic in ggml before having to write ggml logic to import every tensor. You can accomplish this by calling `continue` at the beginning of an iteration of the for loop for a particular tensor if its name is not among the names of the tensors you want to export.

Now, move `./simple-ggml-model.bin` to the `build` folder of ggml. 


## Loading the model into the GGML runtime


### Defining the Model Struct
First of all, we need to define one or more structs to hold all the necessary pointers to keep track of the tensors required by the model. Here is how it's done for the tortoise.cpp autoregressive model:

````c++
// This is a simple model with a weight matrix and bias vector
struct simple_model {

    struct ggml_tensor * weight;
    struct ggml_tensor * bias;


    std::map<std::string, struct ggml_tensor *> tensors;


    // the backend to perform the computation (CPU, CUDA, METAL)
    ggml_backend_t backend = NULL;

    // the backend buffer to storage the tensors data of a and b
    ggml_backend_buffer_t buffer;

    // the context to define the tensor information (dimensions, size, memory address)
    struct ggml_context * ctx;
};
````
You can replace these tensors with your own tensors if you are using this blog post as a template. A useful trick: if you have repeated subcomponents, you can create another struct to hold the repeated tensors, then define a vector of those struct. Here is an [example](https://github.com/balisujohn/tortoise.cpp/blob/b9bb8771c3e3fa8ccb615d24b59b42edf15ca2fa/main.cpp#L365) in tortoise.cpp. 

### Defining the Model Load Process

Next, you'll want to define a `model_load` function to replace the `load_model` function in `simple-backend.cpp`. I've provided an abbreviated version below, the parts included are the parts you'll need to fill in for each tensor you want to import. You need to add to the  `buffer_size` variable the number of bytes for each tensor, which is calculated as the number of numbers contained in the tensor times the size of an element in bytes. For each tensor, you'll also need to declare it with the appropriate shape, and finally add it to the tensors map with the name it was exported with (the name that appears in the output of `convert.py`). Finally, you need to set the number multiplied by `ggml_tensor_overhead()`  to reflect the total number of tensors that will be loaded.  

````c++
  bool simple_model_load(const std::string &fname, simple_model &model) {
  printf("%s: loading model from '%s'\n", __func__, fname.c_str());
  
  ...

  size_t buffer_size = 0;

  buffer_size += 3 * 3  * ggml_type_sizef(GGML_TYPE_F32); // weight
  buffer_size += 3 * ggml_type_sizef(GGML_TYPE_F32); // bias

  printf("%s: ggml tensor size    = %d bytes\n", __func__,
         (int)sizeof(ggml_tensor));
  printf("%s: backend buffer size = %6.2f MB\n", __func__,
         buffer_size / (1024.0 * 1024.0));

  struct ggml_init_params params = {
      ggml_tensor_overhead() * (size_t)(2), // mem size
      NULL,                                   // mem buffer
      true,                                   // no alloc
  };
  
  ...


  auto &ctx = model.ctx;

  model.weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3,3);
  model.bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);

  model.tensors["weight"] = model.weight;
  model.tensors["bias"] = model.bias;

  ...

  return true;
}

````


## Defining the computational graph

Next, we will define the computational graph. In ggml, you need to declare the tensor operations that will be performed before executing them. To get equivalent behavior to our PyTorch module, the graph looks like this:

```` c++
struct ggml_cgraph * build_graph(const simple_model& model) {
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };

    // create a temporally context to build the graph
    struct ggml_context * ctx0 = ggml_init(params0);

    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);


    struct ggml_tensor * input_vector = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 3);

    ggml_set_name(input_vector, "input_vector");

    struct ggml_tensor * result = ggml_mul_mat(ctx0, ggml_cont(ctx0,ggml_transpose(ctx0,model.weight)), input_vector);


    result = ggml_add(ctx0, result, model.bias);

    result = ggml_relu(ctx0, result);

    // build operations nodes
    ggml_build_forward_expand(gf, result);

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

````
Note how there is an `input_vector` tensor but it's value is not set within the computational graph declaration. It can be set to a literal value externally after calling `build_graph` and before starting the execution of the graph. You can check the `compute` function in the complete template to see how `input_vector` is assigned a literal value. 

## Putting it all together

Here is a drop in replacement for `simple-backend.cpp` which will load `./simple-ggml-model.bin` from the build folder that replicates the computations in `simple.py`. I am using commit `a3c0188a4b5d3dec052ff87c9f773baa53631d70` from ggml, so in case the drop in replacement doesn't work in a future GGML you can revert to this commit to get something working before updating to comply with changes to ggml. 

````c++
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

// This is a simple model with a weight matrix and bias vector
struct simple_model {

    struct ggml_tensor * weight;
    struct ggml_tensor * bias;


    std::map<std::string, struct ggml_tensor *> tensors;


    // the backend to perform the computation (CPU, CUDA, METAL)
    ggml_backend_t backend = NULL;

    // the backend buffer to storage the tensors data of a and b
    ggml_backend_buffer_t buffer;

    // the context to define the tensor information (dimensions, size, memory address)
    struct ggml_context * ctx;
};

bool simple_model_load(const std::string &fname, simple_model &model) {
  printf("%s: loading model from '%s'\n", __func__, fname.c_str());

  auto fin = std::ifstream(fname, std::ios::binary);
  if (!fin) {
    fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
    return false;
  }

  // verify magic
  {
    uint32_t magic;
    fin.read((char *)&magic, sizeof(magic));
    if (magic != GGML_FILE_MAGIC) {
      fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__,
              fname.c_str());
      return false;
    }
  }

  size_t buffer_size = 0;

  buffer_size += 3 * 3  * ggml_type_sizef(GGML_TYPE_F32); // weight
  buffer_size += 3 * ggml_type_sizef(GGML_TYPE_F32); // bias


  printf("%s: ggml tensor size    = %d bytes\n", __func__,
         (int)sizeof(ggml_tensor));
  printf("%s: backend buffer size = %6.2f MB\n", __func__,
         buffer_size / (1024.0 * 1024.0));

  struct ggml_init_params params = {
      ggml_tensor_overhead() * (size_t)(2), // mem size
      NULL,                                   // mem buffer
      true,                                   // no alloc
  };

  model.ctx = ggml_init(params);

  if (!model.ctx) {
    fprintf(stderr, "%s: ggml_init() failed\n", __func__);
    return false;
  }

  // initialize the backend
#ifdef GGML_USE_CUBLAS
  fprintf(stderr, "%s: using CUDA backend\n", __func__);
  model.backend = ggml_backend_cuda_init(0);
  if (!model.backend) {
    fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
  }
#endif

#ifdef GGML_USE_METAL
  fprintf(stderr, "%s: using Metal backend\n", __func__);
  ggml_metal_log_set_callback(ggml_log_callback_default, nullptr);
  model.backend = ggml_backend_metal_init();
  if (!model.backend) {
    fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
  }

#endif

  if (!model.backend) {
    // fallback to CPU backend
    fprintf(stderr, "%s: using CPU backend\n", __func__);
    model.backend = ggml_backend_cpu_init();
  }

  if (!model.backend) {
    fprintf(stderr, "%s: ggml_backend_cpu_init() failed\n", __func__);
    return false;
  }


  auto &ctx = model.ctx;

  model.weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3,3);
  model.bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);

  model.tensors["weight"] = model.weight;
  model.tensors["bias"] = model.bias;

  {
    // ggml_allocr * alloc = ggml_allocr_new_from_buffer(model.buffer);
    model.buffer = ggml_backend_alloc_ctx_tensors(ctx, model.backend);

    size_t total_size = 0;

    bool has_lm_head = false;

    std::vector<char> read_buf;

    while (true) {
      int32_t n_dims;
      int32_t length;
      int32_t ttype;

      fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
      fin.read(reinterpret_cast<char *>(&length), sizeof(length));
      fin.read(reinterpret_cast<char *>(&ttype), sizeof(ttype));

      if (fin.eof()) {
        break;
      }

      int32_t nelements = 1;
      int32_t ne[2] = {1, 1};
      for (int i = 0; i < n_dims; ++i) {
        fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
        nelements *= ne[i];
      }

      std::string name(length, 0);
      fin.read(&name[0], length);

      if (model.tensors.find(name) == model.tensors.end()) {
        fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__,
                name.c_str());
        return false;
      }

      auto tensor = model.tensors[name];
      ggml_set_name(tensor, name.c_str());
      if (ggml_nelements(tensor) != nelements) {
        fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n",
                __func__, name.c_str());
        return false;
      }

      if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
        fprintf(stderr,
                "%s: tensor '%s' has wrong shape in model file: got [%d, %d], "
                "expected [%d, %d]\n",
                __func__, name.c_str(), (int)tensor->ne[0], (int)tensor->ne[1],
                ne[0], ne[1]);
        return false;
      }

      // for debugging
      if (0) {
        printf("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n",
               name.c_str(), ne[0], ne[1], ggml_type_name(ggml_type(ttype)),
               ggml_nbytes(tensor) / 1024.0 / 1024.0, ggml_nbytes(tensor));
      }

      const size_t bpe = ggml_type_size(ggml_type(ttype));

      if ((nelements * bpe) / ggml_blck_size(tensor->type) !=
          ggml_nbytes(tensor)) {
        fprintf(stderr,
                "%s: tensor '%s' has wrong size in model file: got %zu, "
                "expected %zu\n",
                __func__, name.c_str(), ggml_nbytes(tensor), nelements * bpe);
        return false;
      }

      if (ggml_backend_buffer_is_host(model.buffer)) {
        // for some backends such as CPU and Metal, the tensor data is in system
        // memory and we can read directly into it
        fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
      } else {
        // read into a temporary buffer first, then copy to device memory
        read_buf.resize(ggml_nbytes(tensor));
        fin.read(read_buf.data(), ggml_nbytes(tensor));
        ggml_backend_tensor_set(tensor, read_buf.data(), 0,
                                ggml_nbytes(tensor));
      }

      total_size += ggml_nbytes(tensor);
    }

    printf("%s: model size  = %8.2f MB\n", __func__,
           total_size / 1024.0 / 1024.0);
  }

  fin.close();

  return true;
}

// build the compute graph
struct ggml_cgraph * build_graph(const simple_model& model) {
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };

    // create a temporally context to build the graph
    struct ggml_context * ctx0 = ggml_init(params0);

    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);


    struct ggml_tensor * input_vector = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 3);

    ggml_set_name(input_vector, "input_vector");

    struct ggml_tensor * result = ggml_mul_mat(ctx0, ggml_cont(ctx0,ggml_transpose(ctx0,model.weight)), input_vector);


    result = ggml_add(ctx0, result, model.bias);

    result = ggml_relu(ctx0, result);

    // build operations nodes
    ggml_build_forward_expand(gf, result);

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

// compute with backend
struct ggml_tensor * compute(const simple_model & model, ggml_gallocr_t allocr) {
    // reset the allocator to free all the memory allocated during the previous inference

    struct ggml_cgraph * gf = build_graph(model);

    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);

    int n_threads = 1; // number of threads to perform some operations with multi-threading

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(model.backend)) {
        ggml_backend_metal_set_n_cb(model.backend, n_threads);
    }
#endif


    std::vector<float> input_data = {1,1,1};

    struct ggml_tensor *input_vector = ggml_graph_get_tensor(gf, "input_vector");

    ggml_backend_tensor_set(input_vector, input_data.data(), 0, input_data.size() * ggml_element_size(input_vector));


    ggml_backend_graph_compute(model.backend, gf);

    // in this case, the output tensor is the last one in the graph
    return gf->nodes[gf->n_nodes - 1];
}

int main(void) {
    ggml_time_init();


    simple_model model;


    std::string simple_model_file_path = "./simple-ggml-model.bin";

    // load the model
    {
        if (!simple_model_load(simple_model_file_path, model)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__,
                simple_model_file_path.c_str());
        exit(1);
        }
    }

    // calculate the temporaly memory required to compute
    ggml_gallocr_t allocr = NULL;

    {
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        // create the worst case graph for memory usage estimation
        struct ggml_cgraph * gf = build_graph(model);
        ggml_gallocr_reserve(allocr, gf);
        size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);

        fprintf(stderr, "%s: compute buffer size: %.4f KB\n", __func__, mem_size/1024.0);
    }


    // perform computation
    struct ggml_tensor * result = compute(model, allocr);

    // create a array to print result
    std::vector<float> out_data(ggml_nelements(result));

    // bring the data from the backend memory
    ggml_backend_tensor_get(result, out_data.data(), 0, ggml_nbytes(result));

    // expected result:
    // [ 60.00 110.00 54.00 29.00
    //  55.00 90.00 126.00 28.00
    //  50.00 54.00 42.00 64.00 ]

    printf("mul mat (%d x %d) (simple result):\n[", (int) result->ne[0], (int) result->ne[1]);
    for (int j = 0; j < result->ne[1] /* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < result->ne[0] /* cols */; i++) {
            printf(" %.2f", out_data[i * result->ne[1] + j]);
        }
    }
    printf(" ]\n");

    // release backend memory used for computation
    ggml_gallocr_free(allocr);

    // free memory
    ggml_free(model.ctx);

    // release backend memory and free backend
    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    return 0;
}
````

When you run this example, you should see the following output:

````
simple_model_load: loading model from './simple-ggml-model.bin'
simple_model_load: ggml tensor size    = 336 bytes
simple_model_load: backend buffer size =   0.00 MB
simple_model_load: using CPU backend
simple_model_load: model size  =     0.00 MB
main: compute buffer size: 0.1562 KB
mul mat (3 x 1) (simple result):
[ 19.00 23.00 27.00 ]
````

You should be able to compile for other backends without changing the code at all.

## Closing Remarks

This process can be applied to *any* PyTorch module consisting of `float32` and `float16` tensors; good luck!

## License

This includes code derived from GGML, which was originally offered under the following license:

MIT License

Copyright (c) 2023-2024 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.




## Feedback

As a final note, if you liked or disliked this blog post,feel free to let me know; my twitter DMs are open! If you have corrections or want to add notes from your own experience, please add them as a PR on the source repository for this blog:

[https://github.com/balisujohn/balisujohn.github.io/](https://github.com/balisujohn/balisujohn.github.io/)
