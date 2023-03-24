package main

import (
	"fmt"
	"os"

	// "golang.org/x/exp/slices"
	// "github.com/x448/float16"

	"github.com/gotzmann/llama.go/llama"
	"github.com/gotzmann/llama.go/ml"
)

/*
https://huggingface.co/docs/transformers/main/model_doc/llama

vocab_size (int, optional, defaults to 32000) — Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the inputs_ids passed when calling LlamaModel

hidden_size (int, optional, defaults to 4096) — Dimension of the hidden representations.

intermediate_size (int, optional, defaults to 11008) — Dimension of the MLP representations.

num_hidden_layers (int, optional, defaults to 32) — Number of hidden layers in the Transformer encoder.

num_attention_heads (int, optional, defaults to 32) — Number of attention heads for each attention layer in the Transformer encoder.

hidden_act (str or function, optional, defaults to "silu") — The non-linear activation function (function or string) in the decoder.

initializer_range (float, optional, defaults to 0.02) — The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rms_norm_eps (float, optional, defaults to 1e-12) — The epsilon used by the rms normalization layers.

use_cache (bool, optional, defaults to True) — Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if config.is_decoder=True.

tie_word_embeddings(bool, optional, defaults to False) — Whether to tie weight embeddings Example —
*/

/*
#include "ggml.h"

#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#include <signal.h>
#endif

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_BOLD          "\x1b[1m"
*/

/*

static bool is_interacting = false;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    printf(ANSI_COLOR_RESET);
    if (signo == SIGINT) {
        if (!is_interacting) {
            is_interacting=true;
        } else {
            _exit(130);
        }
    }
}
#endif

const char * llama_print_system_info(void) {
    static std::string s;

    s  = "";
    s += "AVX = "       + std::to_string(ggml_cpu_has_avx())       + " | ";
    s += "AVX2 = "      + std::to_string(ggml_cpu_has_avx2())      + " | ";
    s += "AVX512 = "    + std::to_string(ggml_cpu_has_avx512())    + " | ";
    s += "FMA = "       + std::to_string(ggml_cpu_has_fma())       + " | ";
    s += "NEON = "      + std::to_string(ggml_cpu_has_neon())      + " | ";
    s += "ARM_FMA = "   + std::to_string(ggml_cpu_has_arm_fma())   + " | ";
    s += "F16C = "      + std::to_string(ggml_cpu_has_f16c())      + " | ";
    s += "FP16_VA = "   + std::to_string(ggml_cpu_has_fp16_va())   + " | ";
    s += "WASM_SIMD = " + std::to_string(ggml_cpu_has_wasm_simd()) + " | ";
    s += "BLAS = "      + std::to_string(ggml_cpu_has_blas())      + " | ";
    s += "SSE3 = "      + std::to_string(ggml_cpu_has_sse3())      + " | ";
    s += "VSX = "       + std::to_string(ggml_cpu_has_vsx())       + " | ";

    return s.c_str();
}

*/

func main() {
	//int main(int argc, char ** argv) {

	////ggml_time_init();
	////const int64_t t_main_start_us = ggml_time_us();

	//var params gptParams//gpt_params params;
	////params.model = "models/llama-7B/ggml-model.bin";

	////if (gpt_params_parse(argc, argv, params) == false) {
	////    return 1;
	////}

	////if (params.n_ctx > 2048) {
	////fmt.Printf("%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
	////    "expect poor results\n", __func__, params.n_ctx);
	////}

	////if (params.seed < 0) {
	////params.seed = time(NULL);
	////}

	//fmt.Printf("\n[main] seed = %d", params.seed)
	/*
	       std::mt19937 rng(params.seed);
	       if (params.prompt.empty()) {
	           params.prompt = gpt_random_prompt(rng);
	       }

	   //    params.prompt = R"(// this function checks if the number n is prime
	   //bool is_prime(int n) {)";

	       int64_t t_load_us = 0;

	       gpt_vocab vocab;*/

	//modelName := "./LLaMA/7B/ggml-model-f16.bin"
	modelName := "./models/7B/ggml-model-fp32.bin"
	model := llama.NewModel()
	vocab := ml.NewVocab()

	// load the model
	if err := llama.LoadModel(modelName, &model, vocab); err != nil {
		fmt.Printf("\n[main] Failed to load model from '%s'", modelName)
		return
	}

	// print system information
	////{
	//fmt.Printf("\nsystem_info: n_threads = %d / %d | %s\n",
	//    params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
	////}

	n_past := uint32(0)

	////int64_t t_sample_us  = 0;
	////int64_t t_predict_us = 0;

	logits := make([]float32, 0)

	// tokenize the prompt
	prompt := "The best programming language to create general AI and profitable ML startup: "
	// Add a space in front of the first character to match OG llama tokenizer behavior
	prompt = " " + prompt
	////std::vector<gpt_vocab::id> embd_inp = ::llama_tokenize(vocab, params.prompt, true);
	embdInp := ml.Tokenize(vocab, prompt, true)
	fmt.Printf("\n\n=== TOKENIZER ===\n\n%+v", embdInp)

	////params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int) embd_inp.size());

	// tokenize the reverse prompt
	////std::vector<gpt_vocab::id> antiprompt_inp = ::llama_tokenize(vocab, params.antiprompt, false);

	fmt.Printf("\nPROMPT = '%s'\n", prompt)
	fmt.Printf("\n#TOKENS = %d\n", len(embdInp))
	for i := 0; i < len(embdInp); i++ {
		fmt.Printf("\n%d => '%s'", embdInp[i], vocab.ID2Token[embdInp[i]])
	}

	////if (params.interactive) {

	////#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
	////        struct sigaction sigint_action;
	////        sigint_action.sa_handler = sigint_handler;
	////        sigemptyset (&sigint_action.sa_mask);
	////        sigint_action.sa_flags = 0;
	////        sigaction(SIGINT, &sigint_action, NULL);
	////#elif defined (_WIN32)
	////       signal(SIGINT, sigint_handler);
	////#endif

	////fmt.Printf("%s: interactive mode on.\n", __func__);

	////if(antiprompt_inp.size()) {
	////    fmt.Printf("%s: reverse prompt: '%s'\n", __func__, params.antiprompt.c_str());
	////    fmt.Printf("%s: number of tokens in reverse prompt = %zu\n", __func__, antiprompt_inp.size());
	////    for (int i = 0; i < (int) antiprompt_inp.size(); i++) {
	////        fmt.Printf("%6d -> '%s'\n", antiprompt_inp[i], vocab.id_to_token.at(antiprompt_inp[i]).c_str());
	////    }
	////    fmt.Printf("\n");
	////}

	////}

	////fmt.Printf("\n\nsampling parameters: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f", params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty)

	var embd []uint32

	fmt.Printf("\n[LLaMM] Start Inference...")

	// determine the required inference memory per token:
	memPerToken := uint32(0)
	fmt.Printf("\nllamaEval #1")
	llama.Eval(&model, 1 /* FIXME n_threads*/, 0 /*&[]uint32{0, 1, 2, 3}*/, embdInp, logits, &memPerToken)
	fmt.Printf("\nllamaEval #1 returned")

	lastNSize := 64 // utils.h // repeat_last_n = 64 // params.repeat_last_n;
	////std::vector<gpt_vocab::id> last_n_tokens(last_n_size);
	lastNTokens := make([]uint32, lastNSize, lastNSize) // FIXME LEN vs CAP
	///std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

	////if (params.interactive) {
	////fmt.Printf("== Running in interactive mode. ==\n"
	////#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
	////              " - Press Ctrl+C to interject at any time.\n"
	////#endif
	////              " - Press Return to return control to LLaMa.\n"
	////              " - If you want to submit another line, end your input in '\\'.\n");
	////   }

	////remainingTokens = params.n_predict
	remainingTokens := uint32(100) // FIXME
	inputConsumed := uint32(0)
	inputNoEcho := false

	// prompt user immediately after the starting prompt has been loaded
	////if (params.interactive_start) {
	////    is_interacting = true;
	////}

	// set the color for the prompt which will be output initially
	////if (params.use_color) {
	////    printf(ANSI_COLOR_YELLOW);
	////}

	for remainingTokens > 0 {

		// predict
		if len(embd) > 0 {
			////const int64_t t_start_us = ggml_time_us();

			fmt.Printf("\nllamaEval #2")
			if err := llama.Eval(&model, 1 /* FIXME params.n_threads*/, n_past, embd, logits, &memPerToken); err != nil {
				fmt.Printf("\n[ERRRO] Failed to predict")
				os.Exit(1)
			}
			fmt.Printf("\nllamaEval #2 returned")

			////t_predict_us += ggml_time_us() - t_start_us;
		}

		n_past += uint32(len(embd))
		embd = []uint32{} ////embd.clear();

		if len(embdInp) <= int(inputConsumed) {

			// out of user input, sample next token
			topK := uint32(40)             // FIXME utils.h // top_k = 40;
			topP := float64(0.95)          // FIXME utils.h // top_p = 0.95f;
			temp := float64(0.80)          // FIXME utils.h // temp  = 0.80f;
			repeatPenalty := float64(1.30) // utils.h // repeat_penalty  = 1.30f;

			vocabSize := 32000 // hparamsVocabSize

			////id := 0

			////{
			////const int64_t t_start_sample_us = ggml_time_us();

			////id = llama_sample_top_p_top_k(vocab, logits.data() + (logits.size() - n_vocab), last_n_tokens, repeat_penalty, top_k, top_p, temp, rng);
			id := llama.SampleTopPTopK(vocab, logits[len(logits)-int(vocabSize):], lastNTokens, repeatPenalty, topK, topP, temp /*, rng*/)

			lastNTokens = lastNTokens[1:] ////last_n_tokens.erase(last_n_tokens.begin());
			lastNTokens = append(lastNTokens, id)

			////t_sample_us += ggml_time_us() - t_start_sample_us;
			////}

			// add it to the context
			embd = append(embd, id)

			// echo this to console
			inputNoEcho = false

			// decrement remaining sampling budget
			remainingTokens--

		} else {

			// some user input remains from prompt or interaction, forward it to processing
			for len(embdInp) > int(inputConsumed) {
				embd = append(embd, embdInp[inputConsumed])
				////lastNTokens.erase(last_n_tokens.begin())
				lastNTokens = lastNTokens[1:]
				lastNTokens = append(lastNTokens, embdInp[inputConsumed])
				inputConsumed++
				if len(embd) > /*params.n_batch*/ 8 { // FIXME utils.h // n_batch = 8; // batch size for prompt processing
					break
				}
			}

			// reset color to default if we there is no pending user input
			////if (!input_noecho && params.use_color && embd_inp.size() == input_consumed) {
			////printf(ANSI_COLOR_RESET);
			////}
		}

		// display text
		if !inputNoEcho {
			//for (auto id : embd) {
			////for (auto id : embd) {
			for _, id := range embd { // FIXME Ordered / Unordered ??
				fmt.Printf("%s", vocab.ID2Token[id])
			}
			////fflush(stdout);
		}

		// in interactive mode, and not currently processing queued inputs;
		// check if we should prompt the user for more
		////if (params.interactive && embd_inp.size() <= input_consumed) {
		// check for reverse prompt
		////    if (antiprompt_inp.size() && std::equal(antiprompt_inp.rbegin(), antiprompt_inp.rend(), last_n_tokens.rbegin())) {
		// reverse prompt found
		////        is_interacting = true;
		////    }
		////    if (is_interacting) {
		// currently being interactive
		////        bool another_line=true;
		////        while (another_line) {
		////            fflush(stdout);
		////            char buf[256] = {0};
		////            int n_read;
		////            if(params.use_color) printf(ANSI_BOLD ANSI_COLOR_GREEN);
		////            if (scanf("%255[^\n]%n%*c", buf, &n_read) <= 0) {
		////                // presumable empty line, consume the newline
		////                std::ignore = scanf("%*c");
		////                n_read=0;
		////            }
		////            if(params.use_color) printf(ANSI_COLOR_RESET);

		////            if (n_read > 0 && buf[n_read-1]=='\\') {
		////                another_line = true;
		////                buf[n_read-1] = '\n';
		////                buf[n_read] = 0;
		////            } else {
		////                another_line = false;
		////                buf[n_read] = '\n';
		////                buf[n_read+1] = 0;
		////            }

		////            std::vector<gpt_vocab::id> line_inp = ::llama_tokenize(vocab, buf, false);
		////            embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

		////            remaining_tokens -= line_inp.size();

		////            input_noecho = true; // do not echo this again
		////        }

		////        is_interacting = false;
		////    }
		////}

		// end of text token
		////if (embd.back() == 2) {
		////fmt.Printf(" [ EOF ]\n");
		////break
		////}
	}

	////#if defined (_WIN32)
	////    signal(SIGINT, SIG_DFL);
	////#endif

	// report timing
	////{
	////    const int64_t t_main_end_us = ggml_time_us();

	////    fmt.Printf("\n\n");
	////    fmt.Printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
	////    fmt.Printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
	////    fmt.Printf("%s:   sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
	////    fmt.Printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us/1000.0f, t_predict_us/1000.0f/n_past);
	////    fmt.Printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
	////}

	////ggml_free(model.ctx);

	////if (params.use_color) {
	////    printf(ANSI_COLOR_RESET);
	////}

	return
}
