
{/* <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.18.0/dist/tf.min.js"></script> */ }


function load_remote_model(url) {

    tf.loadGraphModel(url, {
        fromTFHub: true
    }).then(function (m) {
        console.log('loaded model')
        model = m
        model_shape = model.inputs[0].shape
        console.log('model input shape', model_shape)
        test_input = tf.randomNormal([1, model_shape[1], model_shape[2], model_shape[3]])
    })

}


function generate_tiles(image_dims, box, stride = null) {

    const canvas_shape = image_dims

    if (stride == null)
        stride = { width: box.width, height: box.height }

    const image_columns = parseInt(canvas_shape[0] / stride.width)
    const image_rows = parseInt(canvas_shape[1] / stride.height)

    // const total_frames = image_columns * image_rows

    const tile_dims = []

    for (let i = 0; i < image_rows; i++) {

        for (let j = 0; j < image_columns; j++) {

            tile_dims.push({
                x: j * stride.width, y: i * stride.height,
                height: box.height, width: box.width
            })
        }
    }

    return tile_dims
}


function image_to_tf_matrix() {

}


function image_matrix_tile_extract() {

    // create matrix of images
    const canvas_shape = [500, 500]
    const frame_shape = [50, 50]
    const stride = 50

    const image_columns = parseInt(canvas_shape[0] / stride)
    const image_rows = parseInt(canvas_shape[1] / stride)

    const total_frames = image_columns * image_rows
    console.log('frame dims', image_columns, image_rows, total_frames)


    raw_tensors = []

    for (let i = 0; i < image_columns; i++) {

        for (let j = 0; j < image_rows; j++) {

            let data = ctx.getImageData(j * stride, i * stride, ...frame_shape)
            let frame_tensor = tf.browser.fromPixels(data)
            raw_tensors.push(frame_tensor)
        }
    }

    raw_frames = tf.stack(raw_tensors)

    output_tensor = tf.image.resizeBilinear(raw_frames, [model_shape[1], model_shape[2]])

    return output_tensor
}

function vector_similarity_score(truth_vector, vectors) {

    // MAE
    // return vectors.sub(truth_vector).abs().mean(axis = 1)

    // COSINE similarity
    const dot_product = tf.dot(vectors, truth_vector.reshape([256]))
    const x_length = tf.euclideanNorm(vectors, axis = 1)
    const y_length = truth_vector.euclideanNorm()

    console.log(dot_product.shape, x_length.shape, y_length.shape)

    const cosine_distance = dot_product.div((x_length.mul(y_length)))

    console.log('cos dist', cosine_distance.dataSync())

    return cosine_distance
}



function draw_similarity_map(sim_map) {

    var c2 = document.getElementById("output_canvas");
    var ctx2 = c2.getContext("2d");

    var new_image = ctx2.createImageData(10, 10); // only do this once per page
    var image_data = new_image.data;
    for (let index = 0; index < 100; index++) {

        const offset = index * 4

        image_data[offset] = sim_map[index] * 255;
        image_data[offset + 1] = sim_map[index] * 255;
        image_data[offset + 2] = sim_map[index] * 255;
        image_data[offset + 3] = 255;
    }
    // var d = id.data;

    console.log(new_image)

    ctx2.putImageData(new_image, 0, 0);

    return output_tensor
}


function draw_pixels_to_canvas(images_tensor, index) {

    const single_image = images_tensor.slice([index, 0, 0, 0], [1, images_tensor.shape[1], images_tensor.shape[2], 3]).reshape([images_tensor.shape[1], images_tensor.shape[2], 3]).div(255.)

    tf.browser.toPixels(single_image, kernal_canvas)
}