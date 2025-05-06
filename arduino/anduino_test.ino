
#include <Arduino.h>
const int NUM_ITERATIONS = 1000;  // 更大可以更精细，比如 10000

class MovingAverageFilter {
public:
  MovingAverageFilter(int window_size = 5) 
    : window_size(window_size), sum(0.0), index(0) {
    // 动态分配 values 数组
    values = new float[window_size];
    
    // 初始化 values 数组为零
    for (int i = 0; i < window_size; i++) {
      values[i] = 0.0;
    }
  }

  // 析构函数，释放动态分配的内存
  ~MovingAverageFilter() {
    delete[] values;
  }

  // 更新数据并返回新的平均值
  float update(float new_value) {
    // 更新当前窗口的值
    values[index] = new_value;
    
    // 重新计算总和
    sum = 0.0;
    for (int i = 0; i < window_size; i++) {
      sum += values[i];
    }

    // 计算新的平均值
    index = (index + 1) % window_size;  // 环形缓冲区
    return sum / window_size;
  }

private:
  int window_size;
  float* values;  // 动态分配的数组
  float sum;      // 当前窗口所有值的总和
  int index;      // 当前索引
};


// ==== 卡尔曼滤波器 ====
class KalmanFilter {
public:
  KalmanFilter(float process_noise_Q = 1.0, float measurement_noise_R = 4.0, float estimate_error_P = 2.0, float initial_value = 75.0)
      : Q(process_noise_Q), R(measurement_noise_R), P(estimate_error_P), x(initial_value) {}

  float update(float current_value) {
    P += Q;  // 更新误差协方差
    float K = P / (P + R);  // 卡尔曼增益
    x += K * (current_value - x);  // 更新估计值
    P = (1 - K) * P;  // 更新误差协方差
    return x;
  }

private:
  float Q, R, P, x;
};




// ==== PeakFilter ====
class PeakFilter {
public:
  // 构造函数
  PeakFilter() : prev_hr_3p(75.0) {}  // 默认初始化前一个心率为 75.0

  // 更新函数：输入的是 PPG 信号数组，返回更新后的心率
  float update(float ppg_signal[68]) {
    // 1. 排序并找到前三个最大的功率点
    int top3_indices[3] = {0, 1, 2};
    float top3_powers[3] = {0.0, 0.0, 0.0};
    float top3_frequencies[3] = {0.0, 0.0, 0.0};

    quickSort(ppg_signal, 0, 67);  // 对信号数组进行快速排序

    // 2. 获取前三个最大功率对应的频率和功率幅度
    for (int i = 0; i < 3; i++) {
      top3_indices[i] = i;  // 获取前三个最大点的索引（假设已排序）
      top3_powers[i] = ppg_signal[top3_indices[i]];
      top3_frequencies[i] = top3_indices[i] * 0.1;  // 假设频率是索引的0.1倍（根据具体情况调整）
    }

    // 3. 计算心率候选值
    float hr_candidates[3];
    for (int i = 0; i < 3; i++) {
      hr_candidates[i] = top3_frequencies[i] * 60;  // 转换为心率 BPM
    }

    // 4. 计算加权功率幅度，基于上一时刻的心率进行惩罚
    float weighted_powers[3];
    for (int i = 0; i < 3; i++) {
        if (prev_hr_3p != hr_candidates[i]) {
            weighted_powers[i] = top3_powers[i] / abs(prev_hr_3p - hr_candidates[i]);
        } else {
            weighted_powers[i] = top3_powers[i];
        }
    }


    // 5. 选择加权功率幅度最大的心率
    int max_index = 0;
    float max_weighted_power = weighted_powers[0];
    for (int i = 1; i < 3; i++) {
      if (weighted_powers[i] > max_weighted_power) {
        max_weighted_power = weighted_powers[i];
        max_index = i;
      }
    }

    // 更新前一个心率（用于下次计算）
    prev_hr_3p = hr_candidates[max_index];
    return hr_candidates[max_index];  // 返回新的心率
  }

private:
  float prev_hr_3p;  // 上次的心率，用于惩罚的计算

  // 快速排序算法
  void quickSort(float arr[], int low, int high) {
    if (low < high) {
      int pi = partition(arr, low, high);  // 获取划分点
      quickSort(arr, low, pi - 1);  // 对左半部分进行排序
      quickSort(arr, pi + 1, high);  // 对右半部分进行排序
    }
  }

  // 快速排序的划分函数
  int partition(float arr[], int low, int high) {
    float pivot = arr[high];  // 选取数组最后一个元素作为基准
    int i = (low - 1);  // i 是小于 pivot 的元素的下标
    for (int j = low; j < high; j++) {
      if (arr[j] > pivot) {  // 大于 pivot 的元素
        i++;
        swap(arr[i], arr[j]);
      }
    }
    swap(arr[i + 1], arr[high]);  // 把 pivot 放到正确的位置
    return i + 1;
  }

  // 交换两个元素
  void swap(float& a, float& b) {
    float temp = a;
    a = b;
    b = temp;
  }
};



class OutlierFilter {
public:
  OutlierFilter(int window_size = 10, float threshold_bpm = 30.0, float std_hr_factor = 2.0)
    : window_size(window_size), threshold_bpm(threshold_bpm), std_hr_factor(std_hr_factor), index(0) {
    
    values = new float[window_size];  // 动态分配数组
    for (int i = 0; i < window_size; i++) {
      values[i] = 0.0;
    }
  }

  ~OutlierFilter() {
    delete[] values;  // 释放动态分配的内存
  }

  bool is_outlier(float current_value) {
    if (!initialized) return false;

    // 计算均值
    float sum = 0.0, mean, std_dev = 0.0;
    for (int i = 0; i < window_size; i++) {
      sum += values[i];
    }
    mean = sum / window_size;

    // 计算标准差
    for (int i = 0; i < window_size; i++) {
      std_dev += pow(values[i] - mean, 2);
    }
    std_dev = sqrt(std_dev / window_size);

    float delta = abs(current_value - values[index]);

    if (delta > threshold_bpm || abs(current_value - mean) > std_hr_factor * std_dev) {
      return true;
    }
    return false;
  }

  float update(float current_value) {
    if (!initialized && index >= window_size - 1) {
      initialized = true;
    }

    if (is_outlier(current_value)) {
      return values[index];  // 异常值丢弃
    }

    values[index] = current_value;
    index = (index + 1) % window_size;
    return current_value;
  }

private:
  int window_size;
  float threshold_bpm;
  float std_hr_factor;
  int index;
  float* values;
  bool initialized = false;  // 是否已经填满窗口
};

// ==== 索引滤波器 ====
class IndexFilter {
public:
  IndexFilter() {
    index_use_set = false;
    index_vector_length = 0;
  }
  int update(int index) {
    if (!index_use_set) {
      index_use = index;
      index_use_set = true;
      return index;
    }

    int index_deta = abs(index_use - index);

    if (index_deta <= 1) {
      index_use = index;
      index_vector_length = 0;
    } else {
      if (index_vector_length == 0) {
        index_vector[0] = index;
        index_vector_length = 1;
      } else if (index == index_vector[index_vector_length - 1]) {
        if (index_vector_length < MAX_VECTOR_LEN) {
          index_vector[index_vector_length++] = index;
        }
      } else {
        int last = index_vector[index_vector_length - 1];
        bool same_direction =
            ((last - index_use) > 0 && (index - index_use) > 0) ||
            ((last - index_use) < 0 && (index - index_use) < 0);

        if (same_direction) {
          if (index_vector_length < MAX_VECTOR_LEN) {
            index_vector[index_vector_length++] = index;
          }
        } else {
          index_vector[0] = index;
          index_vector_length = 1;
        }
      }
    }

    if (index_vector_length >= index_deta) {
      index_use = index;
      index_vector_length = 0;
    }

    return index_use;
  }

private:
  int index_use;
  bool index_use_set;
  static const int MAX_VECTOR_LEN = 68;
  int index_vector[MAX_VECTOR_LEN];
  int index_vector_length;
};


// 作用不大
class IndexFilterFast {
public:
  IndexFilterFast() {
    index_use_set = false;
    repeat_count = 0;
  }

  int update(int index) {
    if (!index_use_set) {
      index_use = index;
      index_use_set = true;
      return index;
    }

    int delta = abs(index_use - index);

    if (delta <= 1) {
      index_use = index;
      repeat_count = 0;
      return index;
    }

    int direction = index - index_use;

    if (repeat_count == 0) {
      last_direction = direction > 0 ? 1 : -1;
      repeat_count = 1;
    } else if ((direction > 0 && last_direction > 0) || (direction < 0 && last_direction < 0)) {
      repeat_count++;
    } else {
      last_direction = direction > 0 ? 1 : -1;
      repeat_count = 1;
    }

    if (repeat_count >= delta) {
      index_use = index;
      repeat_count = 0;
    }

    return index_use;
  }

private:
  int index_use;
  bool index_use_set;
  int repeat_count;
  int last_direction;
};




class PeakFilterTest {
public:
  PeakFilterTest() {
    // 初始化或设定默认参数
  }

  // update 方法，接受一个浮动数组，进行滤波处理
  float update(float ppg_signal[68]) {
    // 在此实现滤波算法，这里只是一个示例
    float result = 0.0;
    for (int i = 0; i < 68; i++) {
      result += ppg_signal[i];  // 计算信号总和（仅示例）
    }
    return result / 68.0;  // 返回平均值作为滤波后的结果
  }
};



// ==== 测试运行时间（排除产生随机数的影响）====
template <typename FilterType>
unsigned long testFilter(FilterType &filter, bool isIndexFilter = false) {
  unsigned long totalTime = 0;  // 用于累加每次 update 所消耗的时间
  for (int i = 0; i < NUM_ITERATIONS; i++) {
      float value = random(350, 2500) / 10.0;  // 35.0~250.0
      unsigned long startTime = micros();  // 记录更新前的时间戳
      filter.update(value);
      unsigned long endTime = micros();  // 记录更新后的时间戳
      totalTime += (endTime - startTime);  // 累加每次更新所消耗的时间
  }

  return totalTime;  // 返回总共的运行时间
}



template <typename FilterType>
unsigned long testFilterIndex(FilterType &filter, bool isIndexFilter = false) {
  unsigned long totalTime = 0;  // 用于累加每次 update 所消耗的时间

  for (int i = 0; i < NUM_ITERATIONS; i++) {
    int value = random(0, 69) ;
    unsigned long startTime = micros();  // 记录更新前的时间戳
    filter.update(value);
    unsigned long endTime = micros();  // 记录更新后的时间戳
    totalTime += (endTime - startTime);  // 累加每次更新所消耗的时间
  }

  return totalTime;  // 返回总共的运行时间
}





// ==== 测试运行时间（包括传递 68 长度 PPG 信号）====
template <typename FilterType>
unsigned long testFilterPeak(FilterType &filter) {
  unsigned long totalTime = 0;  // 用于累加每次 update 所消耗的时间

  // 模拟一个68长度的PPG信号
  float ppg_signal[68];
  for (int i = 0; i < 68; i++) {
    ppg_signal[i] = random(0, 10) / 1000.0;
  }
  for (int i = 0; i < NUM_ITERATIONS; i++) {
    unsigned long startTime = micros();  // 记录更新前的时间戳
    filter.update(ppg_signal);  // 传递模拟的PPG信号
    unsigned long endTime = micros();  // 记录更新后的时间戳
    totalTime += (endTime - startTime);  // 累加每次更新所消耗的时间
  }

  return totalTime;  // 返回总共的运行时间
}



void setup() {
  Serial.begin(9600);
  while (!Serial);

  Serial.println("== 滤波器运行时间测试 ==");


  // PeakFilter
  PeakFilter peakFilter;
    // 测试 PeakFilter 的运行时间
  unsigned long t5 = testFilterPeak(peakFilter);
  Serial.print("PeakFilter 平均耗时: ");
  Serial.print((float)t5 , 1);
  Serial.println(" us");


  // MovingAverageFilter
  MovingAverageFilter movAvg(10);
  unsigned long t1 = testFilter(movAvg, false);
  Serial.print("MovingAverage 平均耗时: ");
  Serial.print((float)t1, 1);
  Serial.println(" us");

  // KalmanFilter
  KalmanFilter kf;
  unsigned long t2 = testFilter(kf, false);
  Serial.print("Kalman 平均耗时: ");
  Serial.print((float)t2 , 1);
  Serial.println(" us");

  // OutlierFilter
  OutlierFilter of;
  unsigned long t3 = testFilter(of, false);
  Serial.print("Outlier 平均耗时: ");
  Serial.print((float)t3 , 1);
  Serial.println(" us");

  // IndexFilter
  IndexFilter indexFilter;
  unsigned long t4 = testFilter(indexFilter, true);
  Serial.print("IndexFilter 平均耗时: ");
  Serial.print((float)t4 , 1);
  Serial.println(" us");


}

void loop() {
  // Nothing
}

