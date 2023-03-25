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

	// has to be called once at the start of the program to init ggml stuff
	////ggml_time_init();

	//var params gptParams//gpt_params params;
	////params.model = "models/llama-7B/ggml-model.bin";

	////if (gpt_params_parse(argc, argv, params) == false) {
	////    return 1;
	////}

	////if (params.n_ctx > 2048) {
	////fmt.Printf("%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
	////    "expect poor results\n", __func__, params.n_ctx);
	////}

	////if (params.seed <= 0) {
	////params.seed = time(NULL);
	////}

	//fmt.Printf("\n[main] seed = %d", params.seed)

	////std::mt19937 rng(params.seed);
	////if (params.random_prompt) {
	////params.prompt = gpt_random_prompt(rng);
	////}

	// save choice to use color for later
	// (note for later: this is a slightly awkward choice)
	////con_use_color = params.use_color;

	//    params.prompt = R"(// this function checks if the number n is prime
	//bool is_prime(int n) {)";

	////int64_t t_load_us = 0;

	/* MY

	////gpt_vocab vocab;

	//modelName := "./LLaMA/7B/ggml-model-f16.bin"
	modelName := "./models/7B/ggml-model-fp32.bin"
	model := llama.NewModel()
	vocab := ml.NewVocab()

	// load the model
	if err := llama.LoadModel(modelName, &model, vocab); err != nil {
		fmt.Printf("\n[main] Failed to load model from '%s'", modelName)
		return
	}
	*/

	modelName := "./models/7B/ggml-model-fp32.bin"
	ctx := llama.NewContext()
	lparams := llama.ContextParams{}

	// --- load the model

	////auto lparams = llama_context_default_params();

	////lparams.n_ctx      = params.n_ctx;
	////lparams.n_parts    = params.n_parts;
	////lparams.seed       = params.seed;
	////lparams.f16_kv     = params.memory_f16;
	////lparams.logits_all = params.perplexity;
	////lparams.use_mlock  = params.use_mlock;
	////lparams.embedding  = params.embedding;

	lctx, err := llama.InitFromFile(modelName, &lparams)
	if err != nil {
		fmt.Printf("\n[ERROR] error: failed to load model '%s'", modelName)
		os.Exit(1)
	}

	////if (ctx == NULL) {
	////fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
	////return 1;
	////}

	vocab := lctx.Vocab
	model := lctx.Model

	// print system information
	////{
	//fmt.Printf("\nsystem_info: n_threads = %d / %d | %s\n",
	//    params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
	////}

	// determine the maximum memory usage needed to do inference for the given n_batch and n_predict parameters
	// uncomment the "used_mem" line in llama.cpp to see the results
	////if (params.mem_test) {
	////{
	////const std::vector<llama_token> tmp(params.n_batch, 0);
	////llama_eval(ctx, tmp.data(), tmp.size(), 0, params.n_threads);
	////}

	////{
	////const std::vector<llama_token> tmp = { 0, };
	////llama_eval(ctx, tmp.data(), tmp.size(), params.n_predict - 1, params.n_threads);
	////}

	////llama_print_timings(ctx);
	////llama_free(ctx);

	////return 0;
	////}

	////if (params.perplexity) {
	////perplexity(ctx, params);
	////exit(0);
	////}

	n_past := uint32(0)

	//logits := make([]float32, 0)

	// Add a space in front of the first character to match OG llama tokenizer behavior
	////params.prompt.insert(0, 1, ' ');

	// tokenize the prompt
	prompt := "The best programming language to create general AI and profitable ML startup: "
	// Add a space in front of the first character to match OG llama tokenizer behavior
	prompt = " " + prompt
	////std::vector<gpt_vocab::id> embd_inp = ::llama_tokenize(vocab, params.prompt, true);
	embdInp := ml.Tokenize(vocab, prompt, true)
	fmt.Printf("\n\n=== TOKENIZER ===\n\n%+v", embdInp)

	// tokenize the prompt
	////auto embd_inp = ::llama_tokenize(ctx, params.prompt, true);

	////const int n_ctx = llama_n_ctx(ctx);

	////params.n_predict = std::min(params.n_predict, n_ctx - (int) embd_inp.size());

	// prefix & suffix for instruct mode
	////inpPrefix := ml.Tokenize(vocab, "\n\n### Instruction:\n\n", true)
	////inpSuffix := ml.Tokenize(vocab, "\n\n### Response:\n\n", false)

	// in instruct mode, we inject a prefix and a suffix to each input by the user
	////if params.instruct {
	////params.interactive = true
	////params.antiprompt.push_back("### Instruction:\n\n");
	////}

	// enable interactive mode if reverse prompt is specified
	////if params.antiprompt.size() != 0) {
	////params.interactive = true;
	////}

	////if (params.interactive_start) {
	////params.interactive = true;
	////}

	// determine newline token
	tokenNewline := ml.Tokenize(vocab, "\n", false)
	fmt.Printf("\n NEWLINE = %+v", tokenNewline)

	fmt.Printf("\nPROMPT = '%s'\n", prompt)
	fmt.Printf("\n#TOKENS = %d\n", len(embdInp))
	for i := 0; i < len(embdInp); i++ {
		////////////////////////////////////fmt.Printf("\n%d => '%s'", embdInp[i], vocab.ID2Token[embdInp[i]])
		////llama_token_to_str(ctx, embd_inp[i]));
		fmt.Printf("\n%d => '%s'", embdInp[i], ml.Token2Str(vocab, embdInp[i]))
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

	////if(params.antiprompt.size()) {
	////for (auto antiprompt : params.antiprompt) {
	////fprintf(stderr, "Reverse prompt: '%s'\n", antiprompt.c_str());
	////}
	////}

	////////if (!params.input_prefix.empty()) {
	////fprintf(stderr, "Input prefix: '%s'\n", params.input_prefix.c_str());
	////}
	////}

	////fmt.Printf("\n\nsampling parameters: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f", params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty)

	var embd []uint32

	fmt.Printf("\n[LLaMM] Start Inference...")

	// determine the required inference memory per token:
	memPerToken := uint32(0)
	fmt.Printf("\nllamaEval #1")
	///////////////////////////////////////////////////////llama.Eval(model, 1 /* FIXME n_threads*/, 0 /*&[]uint32{0, 1, 2, 3}*/, embdInp, logits, &memPerToken)
	llama.Eval(model, 1 /*&[]uint32{0, 1, 2, 3}*/, embdInp, logits, &memPerToken)

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
	////is_interacting = params.interactive_start || params.instruct;
	////   }

	inputConsumed := uint32(0)
	inputNoEcho := false

	remainingTokens := uint32(100) // FIXME ////remainingTokens = params.n_predict

	// prompt user immediately after the starting prompt has been loaded
	////if (params.interactive_start) {
	////    is_interacting = true;
	////}

	////#if defined (_WIN32)
	////if (params.use_color) {
	////    // Enable ANSI colors on Windows 10+
	////    unsigned long dwMode = 0;
	////    void* hConOut = GetStdHandle((unsigned long)-11); // STD_OUTPUT_HANDLE (-11)
	////    if (hConOut && hConOut != (void*)-1 && GetConsoleMode(hConOut, &dwMode) && !(dwMode & 0x4)) {
	////        SetConsoleMode(hConOut, dwMode | 0x4); // ENABLE_VIRTUAL_TERMINAL_PROCESSING (0x4)
	////    }
	////}
	////#endif

	// the first thing we will do is to output the prompt, so set color accordingly
	////set_console_state(CONSOLE_STATE_PROMPT);

	if params.Embedding {

		embd = embdInp

		if len(embd) > 0 {
			if llama.Eval(ctx, embd, len(embd), n_past, params.ThreadsCount) {
				fmt.Printf("[HALT] Failed to eval")
				return
			}
		}

		embeddings := llama.GetEmbeddings(ctx)

		// TODO: print / use the embeddings

		if params.UseColor {
			////printf(ANSI_COLOR_RESET);
		}

		return
	}

	for remainingTokens > 0 || params.Interactive {

		// predict
		if len(embd) > 0 {

			fmt.Printf("\nllamaEval #2")
            ////if (llama_eval(ctx, embd.data(), embd.size(), n_past, params.n_threads)) {
                ////fprintf(stderr, "%s : failed to eval\n", __func__);
			if err := llama.Eval(ctx,  embd, len(embd), n_past, params.ThreadsCount); err != nil {
				fmt.Printf("\n[ERROR] Failed to eval")
				os.Exit(1)
			}
			fmt.Printf("\nllamaEval #2 returned")

			////t_predict_us += ggml_time_us() - t_start_us;
		}

		n_past += uint32(len(embd))
		embd = []uint32{} ////embd.clear();

		if len(embdInp) <= int(inputConsumed) && !isInteracting {

			// out of user input, sample next token
			topK := uint32(40)             // FIXME utils.h // top_k = 40;
			topP := float64(0.95)          // FIXME utils.h // top_p = 0.95f;
			temp := float64(0.80)          // FIXME utils.h // temp  = 0.80f;
			repeatPenalty := float64(1.30) // utils.h // repeat_penalty  = 1.30f;

			///////////////////////////////////////////////////////////////vocabSize := 32000 // hparamsVocabSize

			////id := 0

	
            logits := llama_get_logits(ctx);

            if params.ignoreEOS) {
                // set the logit of the eos token to zero to avoid sampling it
                //logits[logits.size() - n_vocab + EOS_TOKEN_ID] = 0;
                // TODO: this does not work of params.logits_all == true
                ////assert(params.perplexity == false);
                logits[llama.TokenEOS()] = 0
            }

			////id = llama_sample_top_p_top_k(vocab, logits.data() + (logits.size() - n_vocab), last_n_tokens, repeat_penalty, top_k, top_p, temp, rng);
			id := llama.SampleTopPTopK(vocab, logits[len(logits)-int(vocabSize):], lastNTokens, repeatPenalty, topK, topP, temp /*, rng*/)

			lastNTokens = lastNTokens[1:] ////last_n_tokens.erase(last_n_tokens.begin());
			lastNTokens = append(lastNTokens, id)

			
            // replace end of text token with newline token when in interactive mode
            if id == llama.TokenEOS() && params.Interactive && !params.Instruct {
                id = llama.TokenNewline.front()
                ////if params.antiprompt.size() != 0) {
                    // tokenize and inject first reverse prompt
                    ////const auto first_antiprompt = ::llama_tokenize(ctx, params.antiprompt.front(), false);
                    ////embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                ////}

			}

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

		}

		// display text
		if !inputNoEcho {
			//for (auto id : embd) {
			////for (auto id : embd) {
			for _, id := range embd { // FIXME Ordered / Unordered ??
				////fmt.Printf("%s", vocab.ID2Token[id])
                fmt.Printf("%s", ml.Token2Str(vocab, id)
			}
			////fflush(stdout);
		}

                // reset color to default if we there is no pending user input
                ////if (!input_noecho && (int)embd_inp.size() == input_consumed) {
                ////    set_console_state(CONSOLE_STATE_DEFAULT);
                ////}

		// in interactive mode, and not currently processing queued inputs;
		// check if we should prompt the user for more
		////if (params.interactive && embd_inp.size() <= input_consumed) {
		// check for reverse prompt

        ////std::string last_output;
        ////for (auto id : last_n_tokens) {
        ////    last_output += llama_token_to_str(ctx, id);
        ////}

 // Check if each of the reverse prompts appears at the end of the output.
 ////for (std::string & antiprompt : params.antiprompt) {
    ////if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos) {
        ////is_interacting = true;
        ////set_console_state(CONSOLE_STATE_USER_INPUT);
        ////fflush(stdout);
        ////break;
    ////}
////}

////if (n_past > 0 && is_interacting) {
    // potentially set color to indicate we are taking user input
    ////set_console_state(CONSOLE_STATE_USER_INPUT);

    ////if (params.instruct) {
        ////input_consumed = embd_inp.size();
        ////embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());

        ////printf("\n> ");
    ////}

    ////std::string buffer;
    ////if (!params.input_prefix.empty()) {
        ////buffer += params.input_prefix;
        ////printf(buffer.c_str());
    /////}

    ////std::string line;
    ////bool another_line = true;
    ////do {
        ////std::getline(std::cin, line);
        ////if (line.empty() || line.back() != '\\') {
            ////another_line = false;
        ////} else {
            ////line.pop_back(); // Remove the continue character
        ////}
        ////buffer += line + '\n'; // Append the line to the result
    ////} while (another_line);

    // done taking input, reset color
    ////set_console_state(CONSOLE_STATE_DEFAULT);

    ////auto line_inp = ::llama_tokenize(ctx, buffer, false);
    ////embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

    ////if (params.instruct) {
        ////embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
    ////}


		////            remaining_tokens -= line_inp.size();

		////            input_noecho = true; // do not echo this again
		////        }

        ////if (n_past > 0) {
		////        is_interacting = false;
\
		////    }
		////}

        // end of text token
        ////if (embd.back() == llama_token_eos()) {
            ////if (params.instruct) {
                ////is_interacting = true;
            ////} else {
                ////fprintf(stderr, " [end of text]\n");
                ////break;
            ////}
        ////}

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        ////if (params.interactive && remaining_tokens <= 0) {
            ////remaining_tokens = params.n_predict;
            ////is_interacting = true;
        ////}
    ////}


	////#if defined (_WIN32)
	////    signal(SIGINT, SIG_DFL);
	////#endif

    ////llama_print_timings(ctx);
    ////llama_free(ctx);

    ////set_console_state(CONSOLE_STATE_DEFAULT);

	return
}
