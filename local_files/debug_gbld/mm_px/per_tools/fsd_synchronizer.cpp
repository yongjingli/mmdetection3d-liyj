#include <time.h>
#include <chrono>
#include <xdds.hpp>
#include <functional>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_map>

#include "services/dds_forwarder/dds_forwarder.h"

#include "gflags/gflags.h"
#include "glog/logging.h"

using namespace xpilot::msg;
namespace df = xpilot::msg::dds_forwarder;
namespace rd = xpilot::msg::rd_statemanage;

constexpr int kMsgSentIntervalMs = 10;
constexpr auto kSendSMGSMsg = false;

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_minloglevel = 0;
  FLAGS_logtostderr = 1;

  xpilot::service::dds_forwarder::DdsForwarder::RegisterTypes();

  std::unique_ptr<dds::domain::DomainParticipant> participant_;
  participant_ = std::make_unique<dds::domain::DomainParticipant>(
      dds::core::QosProvider::Default()->create_participant_from_config("XpilotServiceIfLib::DdsForwarder"));

  std::unique_ptr<dds::pub::DataWriter<df::FsdDetectionStatusMsg>> fsd_detection_writer_;
  fsd_detection_writer_ = std::make_unique<dds::pub::DataWriter<df::FsdDetectionStatusMsg>>(
      rti::pub::find_datawriter_by_name<dds::pub::DataWriter<df::FsdDetectionStatusMsg>>(
          *participant_, "DdsForwarderPublisher::FsdDetectionStatusMsg_writer"));
  auto smgs_writer_ = std::make_unique<dds::pub::DataWriter<rd::StateManagement_Governing_Signal_msg>>(
      rti::pub::find_datawriter_by_name<dds::pub::DataWriter<rd::StateManagement_Governing_Signal_msg>>(
          *participant_, "DdsForwarderPublisher::state_management_writer"));
  ;

  int num_messages = 0;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 10 * 1000);
  df::FsdDetectionStatusMsg fsd_detection_status_msg;
  df::ModelMode model_mode;
  df::EnvMode env_mode;
  rd::StateManagement_Governing_Signal_msg smgs_msg;

  int sleep_ms = 10 * 1000;
  while (true) {
    if (num_messages <= 0) {
      model_mode = df::ModelMode::DRIVING_RUN;
    } else {
      model_mode = static_cast<df::ModelMode>((num_messages % 2) + 1);
      env_mode = static_cast<df::EnvMode>((num_messages % 3) + 1);
      sleep_ms = distribution(generator);
    }
    LOG(INFO) << "Switching to ModelMode " << model_mode << ", EnvMode " << env_mode << " in " << sleep_ms << " ms";
    fsd_detection_status_msg.model_mode(model_mode);
    fsd_detection_status_msg.env_mode(env_mode);
    int auto_st = 0;
    for (int i = 0; i < sleep_ms / kMsgSentIntervalMs; i++) {
      if (kSendSMGSMsg) {
        if (i % 200 == 0) {
          auto_st = auto_st > 6 ? 0 : auto_st + 1;
          LOG(INFO) << "Set auto_st " << auto_st;
        }
        smgs_msg.rdmodulecom_16_state(auto_st);
      }
      if (i % 10 == 0) {
        fsd_detection_writer_->write(fsd_detection_status_msg);
      }
      xpilot_os::this_thread::sleep_for(std::chrono::milliseconds(kMsgSentIntervalMs));
      if (kSendSMGSMsg) {
        smgs_writer_->write(smgs_msg);
      }
    }
    ++num_messages;
  }
}
