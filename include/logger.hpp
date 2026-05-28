#pragma once
#include <iostream>
#include <chrono>
#include <string>
#include <iomanip>
#include <type_traits>

class Logger {
public:
    // C++20: Đo thời gian thực thi chính xác cao bằng High Resolution Clock
    template<typename Func>
    static auto profile(const std::string& task_name, Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        
        if constexpr (std::is_void_v<std::invoke_result_t<Func>>) {
            func();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            std::cout << "  [PERF] " << task_name << " took " << std::fixed << std::setprecision(4) << duration.count() << " ms\n";
        } else {
            auto result = func();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            std::cout << "  [PERF] " << task_name << " took " << std::fixed << std::setprecision(4) << duration.count() << " ms\n";
            return result;
        }
    }

    static void log_info(const std::string& message) {
        std::cout << "[INFO] " << message << "\n";
    }
};
