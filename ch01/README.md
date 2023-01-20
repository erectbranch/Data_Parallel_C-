# 1 Introduction

C++을 이용한 Data parallelism은 현대 heterogeneous system에서 paralllel resource를 사용할 수 있게 해준다. 하나의 C++ application으로도 여러 device의 조합을 이용할 수 있다.

> 예를 들면 GPUs, CPUs, FPGAs(Field Programmable Gate Arrays), ASICs(AI Application-Specific Integrated Circuits) 등이 있다.

> 이 책은 C++과 SYCL을 이용한 data-parallel programming을 다룬다.

SYCL(sickle로 알려져 있다. 줄임말이 아니라 그냥 단어이다.)는 **DPC++**(Data Parallel C++) compiler와 같은 SYCL-aware C++ compiler를 사용한다.

DPC++ compiler는 Intel의 open source compiler project이며, GPU, CPU, 그리고 FPGA device 등을 지원한다. 

> Intel oneAPI toolkit에서 제공하는 commercial version 등 몇 가지 다른 버전도 존재한다.

---

## 1.1 DPC++ complier

DPC++ compiler는 Intel의 해당 깃허브에서 다운로드할 수 있다.

> [DPC++ Complier github](https://github.com/intel/llvm)

---

## 1.2 SYCL program Dissection

우선 간단한 SYCL program 예제를 보자.

```cpp
#include <CL/sycl.hpp>
#include <iostream>
using namespace sycl;

const std::string secret {
    "Ifmmp-!xpsme\"\012J()n!tpssz-!Ebwf/!"
    "J(n!bgsbje!J!dbo(u!ep!uibu/!.!IBM\01"};
const auto sz = secret.size();

int main() {
    queue Q;

    char*result = malloc_shared<char>(sz, Q);
    std::memcpy(result, secret.data(), sz);

    Q.parallel_for(sz,[=](auto&i) {
        result[i] -= 1;
    }).wait();

    std::cout << result << "\n";
    return 0;
}
```

> 모든 STCL construct은 `sycl`이라는 namespace에 담겨 있다.

이를 DPC++ compiler로 compile하면 다음과 같은 출력을 얻는다.

```
Hello, world! (and some additional text left to experience by running it)
```

이제 예제 code의 의미를 알아보자.

- `<CL/sycl.hpp>`: SYCL construct들을 정의하기 위해 필요하다.

- `using namespace sycl`: sycl::를 다시 쓰는 것을 방지한다.

- `queue Q`: 특정 device로 직접 work request하기 위한 queue를 만든다.

- `char*result = malloc_shared<char>(sz, Q);`: device끼리 share하는 data를 allocation한다.

- `Q.parallel_for(sz, [=](auto&i)`: device에 work를 enqueue한다.

- `result[i] -= 1`: device에서 작동하는 code이다.(다른 code는 host(CPU)에서 run한다.) 다시 말해 **kernel** code이다.

  - 참고로 각 문자를 decrement하는 code이다.

kernel code는 secret string을 result string으로 decode하지 않고, 각 문자(character)에 decrement를 적용한다. 이처럼 `parallel_for()`를 사용하면 별다른 작업을 거칠 필요 없이 kernel code를 만들 수 있다.

또한 `parallel_for`가 작업을 한번 queue하면, main program과는 asynchronous하게 작동한다. 

> 하지만 kernel의 결과가 나오는 것을 기다려야 하는 작업도 존재할 수 있다. 특히 Unified Shared Memory와 같은 편리한 feature를 사용하면서 더 크게 번질 수 있는 문제가 되었다.

---

## 1.3 Queues and Actions

queue는 device에 직접 작업을 수행할 수 있게 하는 하나뿐인 connection이다. queue는 action에 따라 두 가지로 분류할 수 있다.

- code to execute: 주로 `single_task`, `parallel_for`, `parallel_for_work_group`로 표현된다.

- memory operation: host와 device 사이의 copy 작업, 혹은 memory initialize를 위해 operation을 채우는 작업을 수행한다.

> memory operation은 자동으로 수행되는 memory 관리보다도 더 control할 필요가 있을 때만 사용한다.

---

## 1.4 Scaling

Scaling이란 (추가적인 computing이 가능할 때) 얼마나 program이 speedup할 수 있는지를 나타낸다. 예로 들어 백 개의 택배를 백 개의 트럭에 각각 실어서, 택배 한 개를 보내는 시간에 다 보낼 수 있다면 가장 이상적인 상황이다. 하지만 실제로 이렇게 되기는 어렵다. 보통 speedup을 제한하는 bottleneck 지점이 존재하기 때문이다.(주로 data가 이동하는 과정에서 생긴다.)

computer 관점에서 보면 위 예시에서 백 개의 트럭은 백 개의 processing core에 해당된다. 하지만 이러한 distributing 과정은 **instantaneous**(동시에 일어나는)하게 일어나지 않는다. 언제나 distribution에는 cost가 발생하며, 이 cost는 얼마나 scaling을 할 수 있는가에 영향을 미친다.

---

## 1.5 Single-Source

program이 single-source일 수 있다는 말은, 동일한 traslation unit(예를 들면 SYCL을 지원하는 C++ compiler)에서 host와 device code를 함께 수행할 수 있는 program이라는 뜻이다.

> 참고로 DPC++와 SYCL program을 위해 host는 C++17의 모든 기능을 지원할 수 있어야 한다. 하지만 device는 모든 C++17 기능을 지원할 필요는 없다.

---

## 1.6 Sharing Devices

한 device에서 둘 혹은 그 이상의 program을 run하고 싶은 경우가 있다. 이런 경우 SYCL이나 DPC++를 사용한 program은 권장되지 않는다. 다른 program이 돌아가고 있는 상황이라면 delay가 생길 가능성이 크다. 

> 사실 CPU에서도 마찬가지로 권장되지 않는다. 너무 많은 active program(예를 들면 mail, browser, virus scanning, video editing 등)을 한번에 run하면 system이 overload될 수 있기 때문이다.

특히 여러 node(CPUs + 장착된 devices)로 구성된 supercomputer에서는, 거의 single application에 한정하여 수행한다. (sharing은 잘 고려되지 않는다.)

정리하면 동일한 시간에 같은 device에서 multiple application을 run하고 있다면, Data Parallel C++ program의 performance에 영향이 있을 수 있다는 점에 유의하자.

---