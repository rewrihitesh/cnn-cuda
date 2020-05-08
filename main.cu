#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "layer.h"

#include <cuda.h>
#include <cstdio>
#include <time.h>
#define BATCH_SIZE 256
static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Define layers of CNN
static Layer ml_input = Layer(0, 0, 28*28);
static Layer ml_c1 = Layer(5*5, 6, 24*24*6);
static Layer ml_s1 = Layer(4*4, 1, 6*6*6);
static Layer ml_f = Layer(6*6*6, 10, 10);

static void learn();
// static unsigned int classify(double data[28][28]);
static void test();
static void forward_pass(double data[28][28],Layer l_input, Layer l_c1, Layer  l_s1, Layer  l_f);
static void back_pass(Layer l_input, Layer l_c1, Layer  l_s1, Layer  l_f);

static inline void loaddata()
{
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}

int main(int argc, const  char **argv)
{
	srand(time(NULL));

	CUresult err = cuInit(0);
	if (err != CUDA_SUCCESS) {
		fprintf(stderr, "CUDA initialisation failed with error code - %d\n", err);
		return 1;
	}

	loaddata();
	learn();
	test();

	return 0;
}

// Forward propagation of a single row in dataset
static void forward_pass(double data[28][28],Layer l_input, Layer l_c1, Layer  l_s1, Layer  l_f)
{
	float input[28][28];

	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			input[i][j] = data[i][j];
		}
	}

	l_input.clear();
	l_c1.clear();
	l_s1.clear();
	l_f.clear();

// 	clock_t start, end;
// 	start = clock();

	l_input.setOutput((float *)input);
	
	fp_preact_c1<<<64, 64>>>((float (*)[28])l_input.output, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight);
	fp_bias_c1<<<64, 64>>>((float (*)[24][24])l_c1.preact, l_c1.bias);
	apply_step_function<<<64, 64>>>(l_c1.preact, l_c1.output, l_c1.O);

	fp_preact_s1<<<64, 64>>>((float (*)[24][24])l_c1.output, (float (*)[6][6])l_s1.preact, (float (*)[4][4])l_s1.weight);
	fp_bias_s1<<<64, 64>>>((float (*)[6][6])l_s1.preact, l_s1.bias);
	apply_step_function<<<64, 64>>>(l_s1.preact, l_s1.output, l_s1.O);

	fp_preact_f<<<64, 64>>>((float (*)[6][6])l_s1.output, l_f.preact, (float (*)[6][6][6])l_f.weight);
	fp_bias_f<<<64, 64>>>(l_f.preact, l_f.bias);
	apply_step_function<<<64, 64>>>(l_f.preact, l_f.output, l_f.O);
	
// 	end = clock();
// 	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Back propagation to update weights
static void back_pass(Layer l_input, Layer l_c1, Layer l_s1, Layer l_f)
{
// 	clock_t start, end;

// 	start = clock();

	bp_weight_f<<<64, 64>>>((float (*)[6][6][6])l_f.d_weight, l_f.d_preact, (float (*)[6][6])l_s1.output);
	bp_bias_f<<<64, 64>>>(l_f.bias, l_f.d_preact);

	bp_output_s1<<<64, 64>>>((float (*)[6][6])l_s1.d_output, (float (*)[6][6][6])l_f.weight, l_f.d_preact);
	bp_preact_s1<<<64, 64>>>((float (*)[6][6])l_s1.d_preact, (float (*)[6][6])l_s1.d_output, (float (*)[6][6])l_s1.preact);
	bp_weight_s1<<<64, 64>>>((float (*)[4][4])l_s1.d_weight, (float (*)[6][6])l_s1.d_preact, (float (*)[24][24])l_c1.output);
	bp_bias_s1<<<64, 64>>>(l_s1.bias, (float (*)[6][6])l_s1.d_preact);

	bp_output_c1<<<64, 64>>>((float (*)[24][24])l_c1.d_output, (float (*)[4][4])l_s1.weight, (float (*)[6][6])l_s1.d_preact);
	bp_preact_c1<<<64, 64>>>((float (*)[24][24])l_c1.d_preact, (float (*)[24][24])l_c1.d_output, (float (*)[24][24])l_c1.preact);
	bp_weight_c1<<<64, 64>>>((float (*)[5][5])l_c1.d_weight, (float (*)[24][24])l_c1.d_preact, (float (*)[28])l_input.output);
	bp_bias_c1<<<64, 64>>>(l_c1.bias, (float (*)[24][24])l_c1.d_preact);
// 	end = clock();
// 	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

__global__ void minibatch(int base ,int N, float *err,Layer *ml_f, Layer *ml_s1, Layer *ml_c1, mnist_data *train_set){
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	// check the index for compatibility
	int idx = base*BATCH_SIZE + pos;
	if(idx > N)
		return;
	
	// create temporary layers for parallelised learning
	cublasHandle_t blas;
	cublasCreate(&blas);
	
	Layer tl_input = Layer(0, 0, 28*28);
	Layer tl_c1 = Layer(5*5, 6, 24*24*6);
	tl_c1.copy_p(ml_c1);
	Layer tl_s1 = Layer(4*4, 1, 6*6*6);
	tl_s1.copy_p(ml_s1);
	Layer tl_f = Layer(6*6*6, 10, 10);
	tl_f.copy_p(ml_f);
	
	float t_err; // temporary error for one sample
	
	forward_pass(train_set[idx].data, tl_input, tl_c1, tl_s1, tl_f);

	tl_f.bp_clear();
	tl_s1.bp_clear();
	tl_c1.bp_clear();

	makeError<<<10, 1>>>(tl_f.d_preact, tl_f.output, train_set[idx].label, 10);
	cublasSnrm2(blas, 10, tl_f.d_preact, 1, &t_err);
	atomicAdd(err,t_err/N);

	back_pass(tl_input, tl_c1, tl_s1, tl_f);
	
	atomicAdd(ml_f->d_weight, (1/N) * tl_f.d_weight);
	atomicAdd(ml_c1->d_weight,(1/N) * tl_c1.d_weight);
	atomicAdd(ml_s1->d_weight,(1/N) * tl_s1.d_weight);
}
static void learn()
{
	float err;
	int iter = 50;
	int N;
	clock_t start, end;
	
	double time_taken = 0.0;

	fprintf(stdout ,"Learning\n");

	while (iter-- != 0) {
		err = 0.0f;

		for (int i = 0; i < (int)train_cnt/BATCH_SIZE+1 ; ++i) {
			float tmp_err = 0.0f;
			int rem = train_cnt - i*BATCH_SIZE;
			if(rem < train_cnt)
				N = rem;
			else
				N = BATCH_SIZE;
			
			ml_f.bp_clear();
			ml_s1.bp_clear();
			ml_c1.bp_clear();

			start = clock();
			
			minibatch <<<BATCH_SIZE, 1>>>(i, N, tmp_err, &ml_f, &ml_s1, &ml_c1, train_set);
			
			apply_grad<<<64, 64>>>(ml_f.weight, ml_f.d_weight, ml_f.M * ml_f.N);
			apply_grad<<<64, 64>>>(ml_s1.weight, ml_s1.d_weight, ml_s1.M * ml_s1.N);
			apply_grad<<<64, 64>>>(ml_c1.weight, ml_c1.d_weight, ml_c1.M * ml_c1.N);
			
			err += tmp_err;
			end = clock();
			time_taken+ = ((double) (end - start)) / CLOCKS_PER_SEC;
		}

		err /= train_cnt;
		fprintf(stdout, "error: %e, time_on_gpu: %lf\n", err, time_taken);

		if (err < threshold) {
			fprintf(stdout, "Training complete, error less than threshold\n\n");
			break;
		}

	}
	
	fprintf(stdout, "\n Time - %lf\n", time_taken);
}


// Returns label of given data (0-9)
__global__ unsigned int classify(double data[28][28])
{
	float res[10];

	forward_pass(data, ml_input, ml_c1, ml_s1, ml_f);

	unsigned int max = 0;

	res=ml_f.output;

	for (int i = 1; i < 10; ++i) {
		if (res[max] < res[i]) {
			max = i;
		}
	}

	return max;
}

// Perform forward propagation of test data
static void test()
{
	int error = 0;

	for (int i = 0; i < test_cnt; ++i) {
		if (classify<<<1,1>>>(test_set[i].data) != test_set[i].label) {
			++error;
		}
	}

	fprintf(stdout, "Error Rate: %.2lf%%\n",
		double(error) / double(test_cnt) * 100.0);
}
