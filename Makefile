TARGET = llama
VERSION = $(shell cat VERSION)
OS = linux
ARCH = amd64
PACKAGE = github.com/gotzmann/$(TARGET)
.DEFAULT_GOAL := build-osx

.PHONY: \
	clean \
	tools \
	test \
	coverage \
	fmt \
	build \
	doc \
	release \

all: tools fmt build test release

print-%:
	@echo $* = $($*)

$(TARGET)-build-linux:
	CGO_ENABLED=0 GOOS=$(OS) GOARCH=$(ARCH) go build -ldflags \
	    "-X $(PACKAGE)/version=$(VERSION)" \
	    -v -o $(CURDIR)/$(TARGET) .

$(TARGET)-build-darwin:
	CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build -ldflags \
	    "-X $(PACKAGE)/version=$(VERSION)" \
	    -v -o $(CURDIR)/$(TARGET) .

build: $(TARGET)-build-linux

build-osx: $(TARGET)-build-darwin

clean:
	rm -f $(PACKAGE)-linux-amd64
	rm -f $(PACKAGE)-darwin-arm64
	rm -f coverage.txt

tools:
	go get github.com/axw/gocov/gocov
	go get github.com/matm/gocov-html
	go get github.com/golangci/golangci-lint/cmd/golangci-lint
	go get github.com/gordonklaus/ineffassign
	go get honnef.co/go/tools/cmd/staticcheck
	go get github.com/client9/misspell/cmd/misspell

test:  test_ineffassign    \
       test_staticcheck    \
       test_misspell       \
       test_govet          \
       test_gotest_race    \
       test_gotest_cover

fmt:
	go fmt ./...

coverage:
	gocov test ./... > $(CURDIR)/coverage.out 2>/dev/null
	gocov report $(CURDIR)/coverage.out
	if test -z "$$CI"; then \
	  gocov-html $(CURDIR)/coverage.out > $(CURDIR)/coverage.html; \
	  if which open &>/dev/null; then \
	    open $(CURDIR)/coverage.html; \
	  fi; \
	fi

test_ineffassign:
	@echo "test: ineffassign"
	@ineffassign pkg/llama/*.go || (echo "ineffassign failed"; exit 1)
	@ineffassign pkg/ml/*.go || (echo "ineffassign failed"; exit 1)
	@echo "test: ok"

test_staticcheck:
	@echo "test: staticcheck"
	@staticcheck pkg/llama/*.go || (echo "staticcheck failed"; exit 1)
	@staticcheck pkg/ml/*.go || (echo "staticcheck failed"; exit 1)
	@echo "test: ok"

test_misspell:
	@echo "test: misspell"
	@misspell pkg/llama/*.go || (echo "misspell failed"; exit 1)
	@misspell pkg/ml/*.go || (echo "misspell failed"; exit 1)
	@echo "test: ok"

test_govet:
	@echo "test: go vet"
	@go vet pkg/llama/*.go|| (echo "go vet failed"; exit 1)
	@go vet pkg/ml/*.go || (echo "go vet failed"; exit 1)
	@echo "test: ok"

test_gosec:
	@echo "test: gosec"
	@gosec pkg/llama/ . || (echo "gosec failed"; exit 1)
	@gosec pkg/ml/ . || (echo "gosec failed"; exit 1)
	@echo "test: ok"

test_gotest_race:
	@echo "test: go test -race"
	@go test -race -coverprofile=coverage.txt -covermode=atomic ./ || (echo "go test -race failed"; exit 1)
	@echo "test: ok"

test_gotest_cover:
	@echo "test: go test -cover"
	@go test -cover ./ || (echo "go test -cover failed"; exit 1)
	@echo "test: ok"

testbadge:
	@echo "Running tests to update readme with badge coverage"
	@go tool cover -func=coverage.out -o=coverage.out
	@gobadge -filename=coverage.out -link https://github.com/gotzmann/llama.go/actions/workflows/coverage.yml

doc:
	@echo "doc: http://localhost:8080/pkg/github.com/gotzmann/llama.go"
	godoc -http=:8080 -index

release:
	@echo "release: $(VERSION)"
	@git tag -a $(VERSION) -m "Release $(VERSION)"
	@git push origin $(VERSION)
	@echo "release: ok"

convert16:
	python3 ./scripts/convert.py ./LLaMA/7B/ 1

convert32:
	python3 ./scripts/convert.py ./LLaMA/7B/ 0

quantize:
	./quantize ~/models/7B/ggml-model-f32.bin ~/models/7B/ggml-model-q4_0.bin 2

int4:
	make -j && ./main -m ./models/7B/ggml-model-q4_0.bin -p "How to create conversational AI:" -n 512

fp16:
	make -j && ./main -m ./models/7B/ggml-model-f16.bin -p "How to create conversational AI:" -n 512

pprof:
	go tool pprof -pdf cpu.pprof > cpu.pdf

