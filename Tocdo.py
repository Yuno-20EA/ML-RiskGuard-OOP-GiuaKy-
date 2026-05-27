#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>

struct ActivityLog {
    std::string activityName;
    double durationMs;
};

void printHistory(const std::vector<ActivityLog>& history) {
    std::cout << "\n================ LICH SU HOAT DONG ================\n";
    if (history.empty()) {
        std::cout << "Chua co lich su hoat dong nao.\n";
        return;
    }
    
    for (size_t i = 0; i < history.size(); ++i) {
        std::cout << i + 1 << ". Hoat dong: [" << history[i].activityName 
                  << "] - Thoi gian thuc thi: " << history[i].durationMs << " ms\n";
    }
    std::cout << "===================================================\n";
}

int main() {
    std::vector<ActivityLog> runHistory;

    std::cout << "Bat dau chuong trinh do toc do va ghi lich su...\n\n";

    {
        std::string actName = "Gia lap tinh toan nhẹ (Sleep 500ms)";
        std::cout << "Dang chay: " << actName << "...\n";

        // Bắt đầu bấm giờ
        auto start = std::chrono::high_resolution_clock::now();

        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;

        std::cout << "-> Hoan thanh! Thoi gian: " << elapsed.count() << " ms\n";

        runHistory.push_back({actName, elapsed.count()});
    }

    {
        std::string actName = "Vong lap tinh tong 10 trieu phan tu";
        std::cout << "\nDang chay: " << actName << "...\n";

        auto start = std::chrono::high_resolution_clock::now();

        long long sum = 0;
        for (int i = 0; i < 10000000; ++i) {
            sum += i;
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        std::cout << "-> Hoan thanh! (Tong = " << sum << ") Thoi gian: " << elapsed.count() << " ms\n";

        runHistory.push_back({actName, elapsed.count()});
    }

    return 0;
}
