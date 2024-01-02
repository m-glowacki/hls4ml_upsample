#ifndef NNET_IMAGE_STREAM_H_
#define NNET_IMAGE_STREAM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include <typeinfo>


namespace nnet {

struct resize_config {
    static const unsigned height = 32;
    static const unsigned width = 32;
    static const unsigned n_chan = 1;
    static const unsigned new_height = 64;
    static const unsigned new_width = 64;
};

template <class data_T, typename CONFIG_T> void resize_nearest(hls::stream<nnet::array<ap_fixed<16, 6>, 1u>> &image, hls::stream<nnet::array<ap_fixed<16, 6>, 64u>> &resized) {
    assert(CONFIG_T::new_height % CONFIG_T::height == 0);
    assert(CONFIG_T::new_width % CONFIG_T::width == 0);
    constexpr unsigned ratio_height = 2; //CONFIG_T::new_height / CONFIG_T::height;
    constexpr unsigned ratio_width = 2; //CONFIG_T::new_width / CONFIG_T::width;

ImageHeight:
    for (unsigned h = 0; h < CONFIG_T::height; h++) {
        #pragma HLS PIPELINE

        data_T data_in_row[CONFIG_T::width];

    ImageWidth:
        for (unsigned i = 0; i < CONFIG_T::width; i++) {
            #pragma HLS UNROLL

            data_T in_data = image.read();

        ImageChan:
            for (unsigned j = 0; j < CONFIG_T::n_chan; j++) {
                #pragma HLS UNROLL

                data_in_row[i][j] = in_data[j];
            }
        }

    ResizeHeight:
        for (unsigned i = 0; i < ratio_height; i++) {
            #pragma HLS UNROLL

        ImageWidth2:
            for (unsigned l = 0; l < CONFIG_T::width; l++) {
                #pragma HLS UNROLL

            ResizeWidth:
                for (unsigned j = 0; j < ratio_width; j++) {
                    #pragma HLS UNROLL

                    data_T out_data;
                    PRAGMA_DATA_PACK(out_data)

                ResizeChan:
                    for (unsigned k = 0; k < CONFIG_T::n_chan; k++) {
                        #pragma HLS UNROLL

                        out_data[k] = data_in_row[l][k];
                    }

                    resized.write(nnet::array<ap_fixed<16, 6>, 64u>());
                }
            }
        }
    }
}

} // namespace nnet

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void transpose_3d(hls::stream<data_T>& data_stream, hls::stream<res_T>& res_stream) {
    static constexpr unsigned depth = CONFIG_T::depth;
    static constexpr unsigned height = CONFIG_T::height;
    static constexpr unsigned width = CONFIG_T::width;

    static constexpr unsigned dim_data[3] = {depth, height, width};
    static constexpr unsigned dim_res[3] = {dim_data[CONFIG_T::perm[0]], dim_data[CONFIG_T::perm[1]],
                                            dim_data[CONFIG_T::perm[2]]};

    int index_data[3] = {0}, index_res[3] = {0};

    for (index_data[0] = 0; index_data[0] < dim_data[0]; index_data[0]++) {
        #pragma HLS LOOP_TRIPCOUNT min=depth max=depth
        #pragma HLS PIPELINE II=1

        for (index_data[1] = 0; index_data[1] < dim_data[1]; index_data[1]++) {
            #pragma HLS LOOP_TRIPCOUNT min=height max=height
            #pragma HLS UNROLL

            for (index_data[2] = 0; index_data[2] < dim_data[2]; index_data[2]++) {
                #pragma HLS LOOP_TRIPCOUNT min=width max=width
                #pragma HLS UNROLL

                index_res[0] = index_data[CONFIG_T::perm[0]];
                index_res[1] = index_data[CONFIG_T::perm[1]];
                index_res[2] = index_data[CONFIG_T::perm[2]];

                res_T output_val = static_cast<res_T>(
                    data_stream.read()); // Assuming data_T and res_T have the same size for simplicity

                res_stream.write(output_val);
            }
        }
    }
}

} // namespace nnet

#endif
