# go-touhou-who

[qwqpap/touhou_guess](https://huggingface.co/qwqpap/touhou_guess) 的Golang移植版本

一个用于识别东方 Project 角色的分类库

## Example

```go
package main

import (
	"encoding/json"
	"fmt"
	"log"

	"go-touhou-who/touhouwho"
)

func main() {
	// 1. 创建服务实例
	tw := touhouwho.New()

	// 2. 初始化
	err := tw.Init(touhouwho.InitOptions{
		ModelPath:     "data/model_complete.onnx",
		TransPath:     "data/trans.json",
		CustomDLLPath: "",  // 留空则默认在程序同级目录下寻找动态库
		ImageSize:     384, // 默认 384
	})
	if err != nil {
		log.Fatalf("Service init failed: %v", err)
	}
	// 确保程序结束时释放资源
	defer tw.Destroy()

	// 3. 执行推理
	imagePath := "./test1.png"
	threshold := float32(0.3)

	res, err := tw.Predict(imagePath, threshold)
	if err != nil {
		log.Fatalf("Predict error: %v", err)
	}

	// 4. 统计高于阈值的数量
	aboveThreshold := 0
	for _, r := range res.Items {
		if r.Threshold {
			aboveThreshold++
		}
	}

	fmt.Printf("\n=== 检测结果 ===\n")
	fmt.Printf("检测到 %d 个角色 (阈值: %.2f)\n\n", aboveThreshold, threshold)

	// 5. 打印 Top 10 (有序数组)
	fmt.Println("Top 10 结果:")
	topN := 10
	if topN > len(res.Items) {
		topN = len(res.Items)
	}
	for i := 0; i < topN; i++ {
		r := res.Items[i]
		status := " "
		if r.Threshold {
			status = "✓"
		}
		fmt.Printf("%d. %s %s: %.6f\n", i+1, status, r.Name, r.Prob)
	}

	// 6. JSON 格式映射 (Map)
	fmt.Println("\n=== JSON格式结果 ===")
	jsonData, _ := json.MarshalIndent(res.ProbMap, "", "  ")
	fmt.Println(string(jsonData))
}
```

## Other

原模型训练使用了训练集 https://huggingface.co/datasets/Preacher-26/touhou-embeddings-dataset

截止 Nov 21, 2025

部分更新作品的角色可能无法识别