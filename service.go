package touhouwho

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"math"
	"os"
	"runtime"
	"sort"
	"sync"

	// 注册常见图片格式解码器，支持 PNG, JPEG, GIF, BMP, WebP, TIFF 等
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"

	"github.com/disintegration/imaging"
	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"
	"github.com/yalue/onnxruntime_go"
)

// InitOptions 初始化配置
type InitOptions struct {
	ModelPath         string // ONNX 模型路径 (必填)
	OnnxModelDataPath string // 外部权重 .data 文件路径 (选填)
	TransPath         string // trans.json 字典路径 (必填)
	CustomDLLPath     string // ONNX Runtime 动态库路径 (选填)
	ImageSize         int    // 输入尺寸，默认 384
}

// PredictionItem 单个角色预测结果
type PredictionItem struct {
	ID        string  `json:"id"`
	Name      string  `json:"name"`
	Prob      float32 `json:"probability"`
	Threshold bool    `json:"above_threshold"`
}

// PredictResult 推理结果结构
type PredictResult struct {
	Items   []PredictionItem   `json:"items"`
	ProbMap map[string]float32 `json:"prob_map"`
}

type characterData struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

type transData struct {
	Characters []characterData `json:"characters"`
}

// Service 东方角色识别服务
type Service struct {
	mu         sync.Mutex
	session    *onnxruntime_go.DynamicAdvancedSession
	characters []string
	charMap    map[string]string
	imageSize  int
	initEnv    bool
}

// New 创建一个新的 TouhouWho 服务实例
func New() *Service {
	return &Service{
		charMap: make(map[string]string),
	}
}

// Init 初始化 ONNX 环境、加载模型和角色字典
func (s *Service) Init(opts InitOptions) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if opts.ImageSize <= 0 {
		opts.ImageSize = 384
	}
	s.imageSize = opts.ImageSize

	if _, err := os.Stat(opts.ModelPath); err != nil {
		return fmt.Errorf("model file not found at %s: %w", opts.ModelPath, err)
	}

	dllPath := opts.CustomDLLPath
	if dllPath == "" {
		switch runtime.GOOS {
		case "windows":
			dllPath = "./onnxruntime.dll"
		case "darwin":
			dllPath = "./libonnxruntime.dylib"
		default:
			dllPath = "./libonnxruntime.so"
		}
	}
	onnxruntime_go.SetSharedLibraryPath(dllPath)

	if err := onnxruntime_go.InitializeEnvironment(); err != nil {
		return fmt.Errorf("failed to initialize ONNX Runtime: %w", err)
	}
	s.initEnv = true

	characters, charMap, err := s.loadTrans(opts.TransPath)
	if err != nil {
		return fmt.Errorf("failed to load translation file: %w", err)
	}
	s.characters = characters
	s.charMap = charMap

	sessionOptions, err := onnxruntime_go.NewSessionOptions()
	if err != nil {
		return fmt.Errorf("failed to create session options: %w", err)
	}
	defer sessionOptions.Destroy()

	session, err := onnxruntime_go.NewDynamicAdvancedSession(
		opts.ModelPath,
		[]string{"input"},
		[]string{"output"},
		sessionOptions,
	)
	if err != nil {
		return fmt.Errorf("failed to create ONNX session: %w", err)
	}
	s.session = session

	return nil
}

// Destroy 释放会话与 ONNX 运行环境
func (s *Service) Destroy() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.session != nil {
		s.session.Destroy()
		s.session = nil
	}

	if s.initEnv {
		_ = onnxruntime_go.DestroyEnvironment()
		s.initEnv = false
	}
	return nil
}

// Predict 根据图片文件路径进行推理预测
func (s *Service) Predict(imagePath string, threshold ...float32) (*PredictResult, error) {
	img, err := imaging.Open(imagePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open image: %w", err)
	}
	return s.PredictImage(img, threshold...)
}

// PredictByBinary 【新增】传入图片二进制字节数组进行推理分类 (支持 PNG, JPEG, GIF, BMP, WebP 等)
func (s *Service) PredictByBinary(imgBytes []byte, threshold ...float32) (*PredictResult, error) {
	if len(imgBytes) == 0 {
		return nil, fmt.Errorf("image binary data is empty")
	}

	// 解码二进制流（imaging.Decode 会自动识别格式并处理 EXIF 方向旋转）
	reader := bytes.NewReader(imgBytes)
	img, err := imaging.Decode(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image binary: %w", err)
	}

	return s.PredictImage(img, threshold...)
}

// PredictImage 支持直接传入 image.Image 对象进行推理
func (s *Service) PredictImage(img image.Image, threshold ...float32) (*PredictResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.session == nil {
		return nil, fmt.Errorf("service is not initialized or has been destroyed")
	}

	th := float32(0.3)
	if len(threshold) > 0 && threshold[0] >= 0 {
		th = threshold[0]
	}

	// 1. 图像预处理
	inputData := s.preprocess(img)

	// 2. 构造 Tensor
	inputShape := onnxruntime_go.NewShape(1, 3, int64(s.imageSize), int64(s.imageSize))
	inputTensor, err := onnxruntime_go.NewTensor(inputShape, inputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// 3. 执行推理
	outputs := []onnxruntime_go.Value{nil}
	if err := s.session.Run([]onnxruntime_go.Value{inputTensor}, outputs); err != nil {
		return nil, fmt.Errorf("session run failed: %w", err)
	}
	defer func() {
		for _, out := range outputs {
			if out != nil {
				out.Destroy()
			}
		}
	}()

	if len(outputs) == 0 || outputs[0] == nil {
		return nil, fmt.Errorf("no output from model")
	}

	outputTensor, ok := outputs[0].(*onnxruntime_go.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected output tensor type: %T", outputs[0])
	}

	rawOutput := outputTensor.GetData()

	// 4. 组装结果
	items := make([]PredictionItem, len(s.characters))
	probMap := make(map[string]float32, len(s.characters))

	for i, charID := range s.characters {
		if i >= len(rawOutput) {
			break
		}
		prob := sigmoid(rawOutput[i])
		name := s.charMap[charID]
		if name == "" {
			name = charID
		}

		item := PredictionItem{
			ID:        charID,
			Name:      name,
			Prob:      prob,
			Threshold: prob >= th,
		}
		items[i] = item
		probMap[name] = prob
	}

	// 按概率降序排序
	sort.Slice(items, func(i, j int) bool {
		return items[i].Prob > items[j].Prob
	})

	return &PredictResult{
		Items:   items,
		ProbMap: probMap,
	}, nil
}

func (s *Service) preprocess(img image.Image) []float32 {
	resized := imaging.Resize(img, s.imageSize, s.imageSize, imaging.Lanczos)
	bounds := resized.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	channelSize := w * h

	pixels := make([]float32, 3*channelSize)
	mean := []float32{0.485, 0.456, 0.406}
	std := []float32{0.229, 0.224, 0.225}

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := resized.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			rFloat := float32(r) / 65535.0
			gFloat := float32(g) / 65535.0
			bFloat := float32(b) / 65535.0

			idx := y*w + x
			pixels[0*channelSize+idx] = (rFloat - mean[0]) / std[0]
			pixels[1*channelSize+idx] = (gFloat - mean[1]) / std[1]
			pixels[2*channelSize+idx] = (bFloat - mean[2]) / std[2]
		}
	}
	return pixels
}

func (s *Service) loadTrans(transPath string) ([]string, map[string]string, error) {
	data, err := os.ReadFile(transPath)
	if err != nil {
		return nil, nil, err
	}

	var td transData
	if err := json.Unmarshal(data, &td); err != nil {
		return nil, nil, err
	}

	chars := make([]string, 0, len(td.Characters))
	cMap := make(map[string]string, len(td.Characters))
	for _, c := range td.Characters {
		chars = append(chars, c.ID)
		cMap[c.ID] = c.Name
	}
	return chars, cMap, nil
}

func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}