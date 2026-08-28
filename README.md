# go-touhou-who

[qwqpap/touhou_guess](https://huggingface.co/qwqpap/touhou_guess) 的 Golang 移植版本。

一个基于 ONNX Runtime、用于识别东方 Project 角色的多标签/单标签图像分类库。

---

## 特性

- 🚀 **高性能推理**：底层基于 ONNX Runtime C-API，支持跨平台（Windows / Linux / macOS）。
- 🖼️ **多格式图像支持**：支持 PNG、JPEG、GIF、BMP、WebP、TIFF 等主流图像格式自动解码。
- 📦 **多种调用方式**：
  - 文件路径推理 (`Predict`)
  - 内存二进制字节流推理 (`PredictByBinary`)，非常适合 Web 服务 / API 网关
  - `image.Image` 原生图像对象推理 (`PredictImage`)
- 📊 **丰富易用的返回数据**：同时返回概率降序排列的列表以及角色名-置信度映射表。

---

## 准备工作

在运行程序之前，需要准备好以下文件：

1. **ONNX 运行时动态库**：
   - Windows: `onnxruntime.dll`
   - Linux: `libonnxruntime.so`
   - macOS: `libonnxruntime.dylib`
   > 可以从 [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases) 下载对应平台的预编译二进制文件，放置在可执行文件同级目录或通过 `CustomDLLPath` 指定路径。
2. **模型文件与词表**：
   - 模型文件：`model.onnx`（若有外挂权重文件，请将 `model.onnx.data` 放置在同目录下）
   - 角色翻译字典：`trans.json`

---

## 安装

```bash
go get github.com/scarletborder/go-touhou-who
```

---

## 如何使用

### 1. 初始化服务

创建一个 `Service` 实例并调用 `Init()` 方法完成 ONNX 会话初始化和模型加载：

```go
tw := touhouwho.New()

err := tw.Init(touhouwho.InitOptions{
    ModelPath:         "data/model.onnx",      // ONNX 模型文件路径（必填）
    TransPath:         "data/trans.json",      // 角色映射字典路径（必填）
    OnnxModelDataPath: "data/model.onnx.data", // 外部权重文件路径（选填）
    CustomDLLPath:     "dlls/onnxruntime.dll", // 动态库文件路径（选填，为空时默认在当前目录查找）
    ImageSize:         384,                    // 输入图像尺寸（选填，默认 384）
})
if err != nil {
    log.Fatalf("初始化失败: %v", err)
}
defer tw.Destroy() // 确保退出时释放会话和 ONNX 环境
```

#### `InitOptions` 参数说明

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `ModelPath` | `string` | **必填**。ONNX 模型文件的物理路径。 |
| `TransPath` | `string` | **必填**。角色 ID 与 Name 对应的 `trans.json` 路径。 |
| `OnnxModelDataPath` | `string` | 选填。若模型外挂权重文件未与 `.onnx` 同名，可显式指定。 |
| `CustomDLLPath` | `string` | 选填。指定 ONNX Runtime 动态链接库的绝对/相对路径。 |
| `ImageSize` | `int` | 选填。模型输入尺寸，默认为 `384`。 |

---

### 2. 角色识别与推理

本库提供了三种不同的推理入口函数，可按需选用：

#### 方式 A：通过本地文件路径识别
```go
threshold := float32(0.3) // 置信度阈值（默认 0.3）
res, err := tw.Predict("path/to/image.png", threshold)
```

#### 方式 B：通过图片二进制流识别（适用于 Web 接口 / 网络下载）
```go
imgBytes, err := os.ReadFile("path/to/image.jpg") // 或从 http.Request Body 读取
res, err := tw.PredictByBinary(imgBytes, 0.3)
```

#### 方式 C：直接传入 `image.Image` 对象
```go
res, err := tw.PredictImage(imgObj, 0.3)
```

---

### 3. 解析返回结果

推理成功后将返回 `PredictResult` 结构体：

```go
type PredictResult struct {
    // Items: 所有角色按概率由大到小排序的切片列表
    Items []PredictionItem
    // ProbMap: 角色名称与对应概率的键值映射表 (map[string]float32)
    ProbMap map[string]float32
}

type PredictionItem struct {
    ID        string  `json:"id"`              // 角色 ID (如 "kochiya_sanae")
    Name      string  `json:"name"`            // 角色名称 (如 "东风谷早苗")
    Prob      float32 `json:"probability"`     // 置信度 (0.0 ~ 1.0)
    Threshold bool    `json:"above_threshold"` // 是否达到预设阈值
}
```

- 获取识别概率最高的第 1 名角色：
  ```go
  top1 := res.Items[0]
  fmt.Printf("最高概率角色: %s (%.2f%%)\n", top1.Name, top1.Prob*100)
  ```
- 快速查询指定角色的概率：
  ```go
  sanaeProb := res.ProbMap["kochiya_sanae"]
  ```

---

## 完整示例

### 示例 1：本地文件路径检测

```go
package main

import (
	"encoding/json"
	"fmt"
	"log"

	"go-touhou-who/touhouwho"
)

func main() {
	tw := touhouwho.New()

	// 1. 初始化
	err := tw.Init(touhouwho.InitOptions{
		ModelPath: "data/model.onnx",
		TransPath: "data/trans.json",
		ImageSize: 384,
	})
	if err != nil {
		log.Fatalf("Service init failed: %v", err)
	}
	defer tw.Destroy()

	// 2. 执行推理
	imagePath := "./data/test1.png"
	threshold := float32(0.3)

	res, err := tw.Predict(imagePath, threshold)
	if err != nil {
		log.Fatalf("Predict error: %v", err)
	}

	// 3. 输出 Top 5 结果
	fmt.Println("Top 5 识别结果:")
	for i := 0; i < 5 && i < len(res.Items); i++ {
		item := res.Items[i]
		tag := "[ ]"
		if item.Threshold {
			tag = "[✓]"
		}
		fmt.Printf("%d. %s %-20s: %.6f\n", i+1, tag, item.Name, item.Prob)
	}
}
```

### 示例 2：使用二进制字节流 (`PredictByBinary`)

```go
package main

import (
	"fmt"
	"log"
	"os"

	"go-touhou-who/touhouwho"
)

func main() {
	tw := touhouwho.New()
	if err := tw.Init(touhouwho.InitOptions{
		ModelPath: "data/model.onnx",
		TransPath: "data/trans.json",
	}); err != nil {
		log.Fatal(err)
	}
	defer tw.Destroy()

	// 读取二进制数据
	imgData, err := os.ReadFile("data/test2.png")
	if err != nil {
		log.Fatal(err)
	}

	// 直接从内存二进制进行推理
	res, err := tw.PredictByBinary(imgData, 0.3)
	if err != nil {
		log.Fatalf("Binary prediction failed: %v", err)
	}

	fmt.Printf("Top 1: %s, 概率: %.4f\n", res.Items[0].Name, res.Items[0].Prob)
}
```

---

## 运行测试

在仓库根目录下执行单元测试：

```bash
go test -v ./...
```

---

## Other

- 原模型训练使用了训练集：[Preacher-26/touhou-embeddings-dataset](https://huggingface.co/datasets/Preacher-26/touhou-embeddings-dataset)
- 数据截止至 2025 年 11 月 21 日，部分更新作品中的新角色可能无法识别。
