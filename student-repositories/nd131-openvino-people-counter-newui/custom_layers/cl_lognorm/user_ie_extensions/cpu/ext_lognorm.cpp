/*
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

// ===============================================================================
// Generated file for Inference Engine extension for CPU plugin
//
// IMPLEMENT YOUR KERNEL HERE.
//
// You need to edit this file in order to:
//  1. initialize parameters (in constructor)
//  2. implement inference logic (in execute() method)
//
// Refer to the section "Adding Your Own Kernels to the Inference Engine" in
// OpenVINO* documentation (either online or offline in
// <INSTALL_DIR>/deployment_tools/documentation/docs/index.html an then navigate
// to the corresponding section).
// ===============================================================================

#include "ext_list.hpp"
#include "ext_base.hpp"
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cassert>
#include <cstdio>
#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#include <immintrin.h>
#endif
#include "ie_parallel.hpp"
#include "tbb/blocked_range.h"

#define DEFINE_PROP(prop_name) \
PropertyVector<float> prop_name;

namespace InferenceEngine {

    class LOGNORMLayer : public CNNLayer {
    
    public:
        /**
         * @brief Axis number for a lognorm operation
         */

        DEFINE_PROP(scale);

        /**
         * @brief assignment operator
         */
        LOGNORMLayer & operator = (const LOGNORMLayer & that) {
            if (&that != this) {
                CNNLayer::operator=(that);
                scale = that.scale;
            }
            return *this;
        }
        /**
         * @brief copy constructor
         */
        LOGNORMLayer(const LOGNORMLayer & that) : CNNLayer(that) {
            operator = (that);
        }
        /**
         * @brief move constructor
         */
        LOGNORMLayer(LOGNORMLayer &&) = default;

        /**
         * @brief Creates a new LOGNORMLayer instance.
         */
        using CNNLayer::CNNLayer;
    };

}

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class LOGNORMImpl: public ExtLayerBase {
public:
    explicit LOGNORMImpl(const CNNLayer* layer) {
        try {
            // LayerSetUp
            // Read parameters from IR and/or initialise them here.
            // Implemented functions for reading parameters are:#

            scale_ = layer->GetParamAsFloat("scale");
            
            // set configuration: specify data format for layer
            // more information about data formats you can find in "Inference Engine Memory primitives" in OpenVINO* documentation
            // (either online or offline in <INSTALL_DIR>/deployment_tools/documentation/docs/index.html an then navigate
            // to the corresponding section). 
            addConfig(layer, { DataConfigurator(ConfLayout::PLN) }, { DataConfigurator(ConfLayout::PLN) });
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                ResponseDesc *resp) noexcept override {
        // Add here implementation for layer inference
        // Examples of implementations you can find in Inerence Engine tool samples/extenstions folder
        
        float* src_data = inputs[0]->buffer();
        float* dst_data = outputs[0]->buffer();

        // Get the dimensions from the input (output dimensions are the same)  
        SizeVector dims = inputs[0]->getTensorDesc().getDims();

        // Get dimensions:N=Batch size, C=Number of Channels, H=Height, W=Width
        int batch_size = static_cast<int>((dims.size() > 0) ? dims[0] : 1);
        int features = static_cast<int>((dims.size() > 1) ? dims[1] : 1);

        std::vector<float> feature_mean(batch_size);
        std::fill(feature_mean.begin(), feature_mean.end(), 0.0f);
        std::vector<float> feature_sum(batch_size);
        std::fill(feature_sum.begin(), feature_sum.end(), 0.0f);
        std::vector<float> feature_min(batch_size);
        std::fill(feature_min.begin(), feature_min.end(), 0.0f);

        std::vector<float> d[batch_size];
        parallel_for(batch_size, [&](int ithr) {
            std::vector<float> v(features);
            parallel_for(features, [&](int fthr) {
                v[fthr] = src_data[ithr*features+fthr];
            });
            d[ithr] = v;
        });

        // calculate mean
        parallel_for(batch_size, [&](int ithr) {
            float sum = 0.0f;
            sum = parallel_sum(0, features, [&](int fthr) {
                return src_data[ithr*features+fthr];
            });
            feature_mean[ithr] = sum / features;
            feature_sum[ithr] = sum;
        });

        // calculate min
        parallel_for(batch_size, [&](int ithr) {
            std::vector<float> v = d[ithr];
            std::vector<float>::iterator result = std::min_element(v.begin(), v.end());
            int idx = std::distance(v.begin(), result);
            feature_min[idx] = v[idx];
        });

        parallel_for(batch_size, [&](int ithr) {
            parallel_for(features, [&](int fthr) {
                src_data[ithr*features+fthr] = scale_* src_data[ithr*features+fthr];
            });
        });

        // Perform (in parallel) the hyperbolic cosine given by: 
        //    lognorm(x) = scale * e^(x-mean) + (x-mean)^2 + 
        //                x.min() * log((x-mean)^2)
        parallel_for2d(batch_size, features, [&](int b, int f) {
            float factor = (std::exp(src_data[b*features+f]-feature_mean[b]) + 
                std::pow(src_data[b*features+f]-feature_mean[b],2.0) + 
                feature_min[b] * std::log(std::pow(src_data[b*features+f]-feature_mean[b],2.0))
            );
            dst_data[b*features+f] = factor;
        });
        return OK;
    }

// attributes of the layer
private:
    float scale_ = 0.0;
};

REG_FACTORY_FOR(ImplFactory<LOGNORMImpl>, LOGNORM);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
