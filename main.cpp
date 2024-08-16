// token_handler.h
#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>

enum class SpecialToken {
    PAD,
    EOS,
    BOS
};

class TokenHandler {
public:
    TokenHandler();
    std::vector<int64_t> tokenize(const std::string& text, bool split_on_space = false);
    std::vector<std::vector<int64_t>> tokenize_batch(const std::vector<std::string>& sentences, bool split_on_space = false);
    std::vector<std::vector<int64_t>> pad_sequences(const std::vector<std::vector<int64_t>>& sequences, int64_t pad_token);
    std::string detokenize(const std::vector<int64_t>& tokens);
    size_t vocab_size() const { return vocab.size(); }

    std::unordered_map<std::string, int64_t> vocab;
    std::unordered_map<int64_t, std::string> inverse_vocab;

private:
    void initialize_vocab();
};

// token_handler.cpp
#include "token_handler.h"
#include <algorithm>

TokenHandler::TokenHandler() {
    initialize_vocab();
}

void TokenHandler::initialize_vocab() {
    std::vector<std::string> special_tokens = {"PAD", "EOS", "BOS"};
    for (size_t i = 0; i < special_tokens.size(); ++i) {
        vocab[special_tokens[i]] = i;
        inverse_vocab[i] = special_tokens[i];
    }

    for (int i = 32; i < 127; ++i) {
        std::string char_str(1, static_cast<char>(i));
        vocab[char_str] = vocab.size();
        inverse_vocab[vocab.size() - 1] = char_str;
    }

    vocab["_"] = vocab.size();
    inverse_vocab[vocab.size() - 1] = "_";
}

std::vector<int64_t> TokenHandler::tokenize(const std::string& text, bool split_on_space) {
    std::vector<int64_t> tokens = {vocab["BOS"]};
    for (char c : text) {
        if (c == ' ') {
            tokens.push_back(vocab["_"]);
        } else if (vocab.find(std::string(1, c)) != vocab.end()) {
            tokens.push_back(vocab[std::string(1, c)]);
        } else {
            tokens.push_back(vocab["PAD"]);
        }

        if ((c == '.' || c == '!' || c == '?') && !split_on_space) {
            tokens.push_back(vocab["EOS"]);
        }
    }

    if (split_on_space) {
        tokens.push_back(vocab["EOS"]);
    }
    return tokens;
}

std::vector<std::vector<int64_t>> TokenHandler::tokenize_batch(const std::vector<std::string>& sentences, bool split_on_space) {
    std::vector<std::vector<int64_t>> result;
    for (const auto& sentence : sentences) {
        result.push_back(tokenize(sentence, split_on_space));
    }
    return result;
}

std::vector<std::vector<int64_t>> TokenHandler::pad_sequences(const std::vector<std::vector<int64_t>>& sequences, int64_t pad_token) {
    size_t max_length = 0;
    for (const auto& seq : sequences) {
        max_length = std::max(max_length, seq.size());
    }

    std::vector<std::vector<int64_t>> padded_sequences;
    for (const auto& seq : sequences) {
        std::vector<int64_t> padded_seq = seq;
        padded_seq.resize(max_length, pad_token);
        padded_sequences.push_back(padded_seq);
    }
    return padded_sequences;
}

std::string TokenHandler::detokenize(const std::vector<int64_t>& tokens) {
    std::string result;
    for (int64_t token : tokens) {
        if (token != vocab["BOS"] && token != vocab["EOS"] && token != vocab["PAD"]) {
            if (inverse_vocab[token] == "_") {
                result += " ";
            } else {
                result += inverse_vocab[token];
            }
        }
    }
    return result;
}

// hierarchical_sentence_transformer.h
#pragma once

#include <torch/torch.h>
#include "token_handler.h"

struct Config {
    int batch_size = 2;
    int max_sequence_length = 512;
    int d_model = 768;
    int nhead = 12;
    int num_layers = 6;
    int dim_feedforward = 3072;
    double dropout = 0.1;
};

class HierarchicalSentenceTransformer : public torch::nn::Module {
public:
    HierarchicalSentenceTransformer(const Config& config, const TokenHandler& token_handler);
    torch::Tensor forward(torch::Tensor src, torch::Tensor mask);

private:
    Config config;
    TokenHandler token_handler;
    torch::nn::Embedding embedding;
    torch::nn::TransformerEncoder transformer;
    torch::nn::Linear fc_out;
};

// hierarchical_sentence_transformer.cpp
#include "hierarchical_sentence_transformer.h"

HierarchicalSentenceTransformer::HierarchicalSentenceTransformer(const Config& config, const TokenHandler& token_handler)
    : config(config),
      token_handler(token_handler),
      embedding(torch::nn::Embedding(token_handler.vocab_size(), config.d_model)),
      transformer(torch::nn::TransformerEncoder(
          torch::nn::TransformerEncoderLayer(
              torch::nn::TransformerEncoderLayerOptions(config.d_model, config.nhead)
                  .dim_feedforward(config.dim_feedforward)
                  .dropout(config.dropout)
                  .activation(torch::kGELU)
                  .batch_first(true)
                  .norm_first(true)),
          config.num_layers)),
      fc_out(torch::nn::Linear(config.d_model, token_handler.vocab_size())) {
    register_module("embedding", embedding);
    register_module("transformer", transformer);
    register_module("fc_out", fc_out);
}

torch::Tensor HierarchicalSentenceTransformer::forward(torch::Tensor src, torch::Tensor mask) {
    auto embedded = embedding(src);
    auto transformer_out = transformer(embedded, mask);
    auto output = fc_out(transformer_out);
    return output;
}

// main.cpp
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "hierarchical_sentence_transformer.h"

TEST_CASE("TokenHandler functionality", "[token_handler]") {
    TokenHandler token_handler;

    SECTION("Tokenize and detokenize") {
        std::string sentence = "The quick brown fox.";
        auto tokens = token_handler.tokenize(sentence);
        auto detokenized = token_handler.detokenize(tokens);
        REQUIRE(detokenized == sentence);
    }

    SECTION("Batch tokenize and pad") {
        std::vector<std::string> sentences = {
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs."
        };
        auto tokenized = token_handler.tokenize_batch(sentences, true);
        auto padded = token_handler.pad_sequences(tokenized, token_handler.vocab["PAD"]);
        REQUIRE(padded.size() == 2);
        REQUIRE(padded[0].size() == padded[1].size());
    }
}

TEST_CASE("HierarchicalSentenceTransformer functionality", "[transformer]") {
    Config config;
    TokenHandler token_handler;
    HierarchicalSentenceTransformer model(config, token_handler);

    SECTION("Forward pass") {
        std::vector<std::string> sentences = {
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs."
        };
        auto tokenized = token_handler.tokenize_batch(sentences, true);
        auto padded = token_handler.pad_sequences(tokenized, token_handler.vocab["PAD"]);
        
        torch::Tensor input_tensor = torch::tensor(padded, torch::kLong);
        torch::Tensor mask = input_tensor != token_handler.vocab["PAD"];

        torch::NoGradGuard no_grad;
        auto output = model.forward(input_tensor, mask);

        REQUIRE(output.sizes() == std::vector<int64_t>{2, padded[0].size(), token_handler.vocab_size()});
    }
}