#/bin/bash

if [ -z "$2" ]; then
    echo "usage: $0 <output-dir> <mnt-dir>"
    exit 1
fi

mkdir -p "$1"
mkdir -p "$2"

OUT=$(realpath "$1")
MNT=$(realpath "$2")

rm -v $OUT/*.log
rm -v $OUT/*.exit
rm -v $OUT/*.md

sd=`dirname $0`
cd $sd/../
SRC=`pwd`

## helpers

# download a file if it does not exist or if it is outdated
function gg_wget {
    local out=$1
    local url=$2

    local cwd=`pwd`

    mkdir -p $out
    cd $out

    # should not re-download if file is the same
    wget -nv -N $url

    cd $cwd
}

function gg_printf {
    printf -- "$@" >> $OUT/README.md
}

function gg_run {
    ci=$1

    set -o pipefail
    set -x

    gg_run_$ci | tee $OUT/$ci.log
    cur=$?
    echo "$cur" > $OUT/$ci.exit

    set +x
    set +o pipefail

    gg_sum_$ci

    ret=$((ret | cur))
}

## ci

# ctest_debug

function gg_run_ctest_debug {
    cd ${SRC}

    rm -rf build-ci-debug && mkdir build-ci-debug && cd build-ci-debug

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Debug ..     ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time make -j                               ) 2>&1 | tee -a $OUT/${ci}-make.log

    (time ctest --output-on-failure -E test-opt ) 2>&1 | tee -a $OUT/${ci}-ctest.log

    set +e
}

function gg_sum_ctest_debug {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs ctest in debug mode\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-ctest.log)"
    gg_printf '```\n'
    gg_printf '\n'
}

# ctest_release

function gg_run_ctest_release {
    cd ${SRC}

    rm -rf build-ci-release && mkdir build-ci-release && cd build-ci-release

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Release ..   ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time make -j                               ) 2>&1 | tee -a $OUT/${ci}-make.log

    if [ -z $GG_BUILD_LOW_PERF ]; then
        (time ctest --output-on-failure ) 2>&1 | tee -a $OUT/${ci}-ctest.log
    else
        (time ctest --output-on-failure -E test-opt ) 2>&1 | tee -a $OUT/${ci}-ctest.log
    fi

    set +e
}

function gg_sum_ctest_release {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs ctest in release mode\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-ctest.log)"
    gg_printf '```\n'
}

# open_llama_3b_v2

function gg_run_open_llama_3b_v2 {
    cd ${SRC}

    gg_wget models-mnt/open-llama/3B-v2/ https://huggingface.co/openlm-research/open_llama_3b_v2/raw/main/config.json
    gg_wget models-mnt/open-llama/3B-v2/ https://huggingface.co/openlm-research/open_llama_3b_v2/resolve/main/tokenizer.model
    gg_wget models-mnt/open-llama/3B-v2/ https://huggingface.co/openlm-research/open_llama_3b_v2/raw/main/tokenizer_config.json
    gg_wget models-mnt/open-llama/3B-v2/ https://huggingface.co/openlm-research/open_llama_3b_v2/raw/main/special_tokens_map.json
    gg_wget models-mnt/open-llama/3B-v2/ https://huggingface.co/openlm-research/open_llama_3b_v2/resolve/main/pytorch_model.bin
    gg_wget models-mnt/open-llama/3B-v2/ https://huggingface.co/openlm-research/open_llama_3b_v2/raw/main/generation_config.json

    gg_wget models-mnt/wikitext/ https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
    unzip -o models-mnt/wikitext/wikitext-2-raw-v1.zip -d models-mnt/wikitext/
    head -n 60 models-mnt/wikitext/wikitext-2-raw/wiki.test.raw > models-mnt/wikitext/wikitext-2-raw/wiki.test-60.raw

    path_models="../models-mnt/open-llama/3B-v2"
    path_wiki="../models-mnt/wikitext/wikitext-2-raw"

    rm -rf build-ci-release && mkdir build-ci-release && cd build-ci-release

    set -e

    (time cmake -DCMAKE_BUILD_TYPE=Release -DLLAMA_QKK_64=1 .. ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time make -j                                              ) 2>&1 | tee -a $OUT/${ci}-make.log

    python3 ../convert.py ${path_models}

    model_f16="${path_models}/ggml-model-f16.bin"
    model_q8_0="${path_models}/ggml-model-q8_0.bin"
    model_q4_0="${path_models}/ggml-model-q4_0.bin"
    model_q4_1="${path_models}/ggml-model-q4_1.bin"
    model_q5_0="${path_models}/ggml-model-q5_0.bin"
    model_q5_1="${path_models}/ggml-model-q5_1.bin"
    model_q3_k="${path_models}/ggml-model-q3_k.bin"
    model_q4_k="${path_models}/ggml-model-q4_k.bin"
    model_q5_k="${path_models}/ggml-model-q5_k.bin"
    model_q6_k="${path_models}/ggml-model-q6_k.bin"

    wiki_test_60="${path_wiki}/wiki.test-60.raw"

    ./bin/quantize ${model_f16} ${model_q8_0} q8_0
    ./bin/quantize ${model_f16} ${model_q4_0} q4_0
    ./bin/quantize ${model_f16} ${model_q4_1} q4_1
    ./bin/quantize ${model_f16} ${model_q5_0} q5_0
    ./bin/quantize ${model_f16} ${model_q5_1} q5_1
    ./bin/quantize ${model_f16} ${model_q3_k} q3_k
    ./bin/quantize ${model_f16} ${model_q4_k} q4_k
    ./bin/quantize ${model_f16} ${model_q5_k} q5_k
    ./bin/quantize ${model_f16} ${model_q6_k} q6_k

    (time ./bin/main --model ${model_f16}  -s 1234 -n 64 -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-f16.log
    (time ./bin/main --model ${model_q8_0} -s 1234 -n 64 -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q8_0.log
    (time ./bin/main --model ${model_q4_0} -s 1234 -n 64 -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q4_0.log
    (time ./bin/main --model ${model_q4_1} -s 1234 -n 64 -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q4_1.log
    (time ./bin/main --model ${model_q5_0} -s 1234 -n 64 -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q5_0.log
    (time ./bin/main --model ${model_q5_1} -s 1234 -n 64 -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q5_1.log
    (time ./bin/main --model ${model_q3_k} -s 1234 -n 64 -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q3_k.log
    (time ./bin/main --model ${model_q4_k} -s 1234 -n 64 -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q4_k.log
    (time ./bin/main --model ${model_q5_k} -s 1234 -n 64 -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q5_k.log
    (time ./bin/main --model ${model_q6_k} -s 1234 -n 64 -p "I believe the meaning of life is" ) 2>&1 | tee -a $OUT/${ci}-tg-q6_k.log

    (time ./bin/perplexity --model ${model_f16}  -f ${wiki_test_60} -c 128 -b 128 --chunks 3 ) 2>&1 | tee -a $OUT/${ci}-tg-f16.log
    (time ./bin/perplexity --model ${model_q8_0} -f ${wiki_test_60} -c 128 -b 128 --chunks 3 ) 2>&1 | tee -a $OUT/${ci}-tg-q8_0.log
    (time ./bin/perplexity --model ${model_q4_0} -f ${wiki_test_60} -c 128 -b 128 --chunks 3 ) 2>&1 | tee -a $OUT/${ci}-tg-q4_0.log
    (time ./bin/perplexity --model ${model_q4_1} -f ${wiki_test_60} -c 128 -b 128 --chunks 3 ) 2>&1 | tee -a $OUT/${ci}-tg-q4_1.log
    (time ./bin/perplexity --model ${model_q5_0} -f ${wiki_test_60} -c 128 -b 128 --chunks 3 ) 2>&1 | tee -a $OUT/${ci}-tg-q5_0.log
    (time ./bin/perplexity --model ${model_q5_1} -f ${wiki_test_60} -c 128 -b 128 --chunks 3 ) 2>&1 | tee -a $OUT/${ci}-tg-q5_1.log
    (time ./bin/perplexity --model ${model_q3_k} -f ${wiki_test_60} -c 128 -b 128 --chunks 3 ) 2>&1 | tee -a $OUT/${ci}-tg-q3_k.log
    (time ./bin/perplexity --model ${model_q4_k} -f ${wiki_test_60} -c 128 -b 128 --chunks 3 ) 2>&1 | tee -a $OUT/${ci}-tg-q4_k.log
    (time ./bin/perplexity --model ${model_q5_k} -f ${wiki_test_60} -c 128 -b 128 --chunks 3 ) 2>&1 | tee -a $OUT/${ci}-tg-q5_k.log
    (time ./bin/perplexity --model ${model_q6_k} -f ${wiki_test_60} -c 128 -b 128 --chunks 3 ) 2>&1 | tee -a $OUT/${ci}-tg-q6_k.log

    function check_ppl {
        qnt="$1"
        ppl=$(echo "$2" | grep -oE "[0-9]+\.[0-9]+" | tail -n 1)

        if [ $(echo "$ppl > 20.0" | bc) -eq 1 ]; then
            printf '  - %s @ %s (FAIL: ppl > 20.0)\n' "$qnt" "$ppl"
            return 20
        fi

        printf '  - %s @ %s OK\n' "$qnt" "$ppl"
        return 0
    }

    check_ppl "f16"  "$(cat $OUT/${ci}-tg-f16.log  | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q8_0" "$(cat $OUT/${ci}-tg-q8_0.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q4_0" "$(cat $OUT/${ci}-tg-q4_0.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q4_1" "$(cat $OUT/${ci}-tg-q4_1.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q5_0" "$(cat $OUT/${ci}-tg-q5_0.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q5_1" "$(cat $OUT/${ci}-tg-q5_1.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q3_k" "$(cat $OUT/${ci}-tg-q3_k.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q4_k" "$(cat $OUT/${ci}-tg-q4_k.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q5_k" "$(cat $OUT/${ci}-tg-q5_k.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log
    check_ppl "q6_k" "$(cat $OUT/${ci}-tg-q6_k.log | grep "^\[1\]")" | tee -a $OUT/${ci}-ppl.log

    set +e
}

function gg_sum_open_llama_3b_v2 {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'OpenLLaMA 3B-v2:\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '- perplexity:\n%s\n' "$(cat $OUT/${ci}-ppl.log)"
    gg_printf '- f16: \n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-f16.log)"
    gg_printf '- q8_0:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q8_0.log)"
    gg_printf '- q4_0:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q4_0.log)"
    gg_printf '- q4_1:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q4_1.log)"
    gg_printf '- q5_0:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q5_0.log)"
    gg_printf '- q5_1:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q5_1.log)"
    gg_printf '- q3_k:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q3_k.log)"
    gg_printf '- q4_k:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q4_k.log)"
    gg_printf '- q5_k:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q5_k.log)"
    gg_printf '- q6_k:\n```\n%s\n```\n' "$(cat $OUT/${ci}-tg-q6_k.log)"
}

## main

if [ -z $GG_BUILD_LOW_PERF ]; then
    rm -rf ${SRC}/models-mnt

    mnt_models=$(realpath ${MNT}/models)
    mkdir -p ${mnt_models}
    ln -sfn ${mnt_models} ${SRC}/models-mnt

    python3 -m pip install -r ${SRC}/requirements.txt
fi

ret=0

#test $ret -eq 0 && gg_run ctest_debug
#test $ret -eq 0 && gg_run ctest_release

if [ -z $GG_BUILD_LOW_PERF ]; then
    test $ret -eq 0 && gg_run open_llama_3b_v2
fi

exit $ret
