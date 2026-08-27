package touhouwho_test

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	touhouwho "go-touhou-who"
)

// findExistingPath 辅助函数：按顺序查找可用的文件路径（兼容从根目录或子包运行 go test）
func findExistingPath(candidates ...string) string {
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return candidates[0]
}

func TestTouhouWhoPredict(t *testing.T) {
	// 1. 定位相关模型、字典及动态库文件
	modelPath := findExistingPath(
		"data/model.onnx",
	)
	transPath := findExistingPath(
		"data/trans.json",
	)
	customDLLPath := findExistingPath(
		"dlls/onnxruntime.dll",
	)

	// 2. 初始化服务
	tw := touhouwho.New()
	err := tw.Init(touhouwho.InitOptions{
		ModelPath:     modelPath,
		TransPath:     transPath,
		CustomDLLPath: customDLLPath,
		ImageSize:     384,
	})
	if err != nil {
		t.Fatalf("Service init failed: %v", err)
	}
	defer func() {
		if err := tw.Destroy(); err != nil {
			t.Errorf("Service destroy error: %v", err)
		}
	}()

	// 3. 测试用例定义
	tests := []struct {
		name         string
		imagePath    string
		expectedTop  string  // 预期第一名的角色 ID / Name
		expectedProb float32 // 预期概率
		tolerance    float32 // 允许的浮点误差范围
	}{
		{
			name: "Test Case 1: Sanae Kochiya",
			imagePath: findExistingPath(
				"data/test1.png",
				"../data/test1.png",
				"test1.png",
				"./test1.png",
			),
			expectedTop:  "kochiya_sanae",
			expectedProb: 0.999976,
			tolerance:    0.1,
		},
		{
			name: "Test Case 2: Kosuzu Motoori",
			imagePath: findExistingPath(
				"data/test2.png",
				"../data/test2.png",
				"test2.png",
				"./test2.png",
			),
			expectedTop:  "motoori_kosuzu",
			expectedProb: 0.982767,
			tolerance:    0.1,
		},
	}

	// 4. 循环执行推理并断言
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// 确保测试图片存在
			if _, err := os.Stat(tc.imagePath); err != nil {
				t.Fatalf("Test image not found at %s: %v", tc.imagePath, err)
			}

			// 执行预测（默认阈值 0.3）
			res, err := tw.Predict(tc.imagePath, 0.3)
			if err != nil {
				t.Fatalf("Prediction failed for %s: %v", filepath.Base(tc.imagePath), err)
			}

			if len(res.Items) == 0 {
				t.Fatal("Prediction result is empty")
			}

			// 获取最高预测项
			topItem := res.Items[0]
			t.Logf("[%s] Top 1 Result -> Name/ID: %s, Prob: %.6f, Threshold: %v",
				filepath.Base(tc.imagePath), topItem.Name, topItem.Prob, topItem.Threshold)

			// 断言 1: 第一名角色匹配 (匹配 ID 或 Name)
			if topItem.ID != tc.expectedTop && topItem.Name != tc.expectedTop {
				t.Errorf("Top prediction mismatch! Expected: %s, got: %s (ID: %s)",
					tc.expectedTop, topItem.Name, topItem.ID)
			}

			// 断言 2: 概率在预期误差范围内
			diff := math.Abs(float64(topItem.Prob - tc.expectedProb))
			if diff > float64(tc.tolerance) {
				t.Errorf("Probability mismatch! Expected ~%.6f, got: %.6f (diff: %.6f > %.6f)",
					tc.expectedProb, topItem.Prob, diff, tc.tolerance)
			}

			// 断言 3: ProbMap 映射数据一致性
			mapKey := topItem.Name
			if mapVal, ok := res.ProbMap[mapKey]; !ok || mapVal != topItem.Prob {
				t.Errorf("ProbMap integrity check failed! Key %s has value %.6f, expected %.6f",
					mapKey, mapVal, topItem.Prob)
			}
		})
	}
}
