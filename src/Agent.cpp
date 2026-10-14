#include "Agent.h"
#include "repast_hpc/Moore2DGridQuery.h"
#include "repast_hpc/Point.h"
#include <stdio.h>

Agent::Agent(repast::AgentId id, int type, double threshold)
    : agentId(id), agentType(type),
      isSatisfied(false), threshold(threshold) {
}

Agent::Agent(repast::AgentId id, int type,
             bool satisfied, double threshold)
    : agentId(id), agentType(type),
      isSatisfied(satisfied), threshold(threshold) {
}

void Agent::set(int currentRank, int newType,
                bool newSatisfiedStatus) {
    agentId.currentRank(currentRank);
    agentType = newType;
    isSatisfied = newSatisfiedStatus;
}

Agent::~Agent() {}


void Agent::updateStatus(repast::SharedContext<Agent>* context,
		repast::SharedDiscreteSpace<Agent, repast::StrictBorders, repast::SimpleAdder<Agent> >* space){
	std::vector<Agent*> neighbourAgents;

// 获取自己的位置
std::vector<int> agentLoc;
space->getLocation(agentId, agentLoc);

// 获取周围八个方向的邻居
repast::Point<int> center(agentLoc);
repast::Moore2DGridQuery<Agent> moore2DQuery(space);
moore2DQuery.query(center, 1, false, neighbourAgents);

// 统计同类邻居
int similarCount = 0;
std::vector<Agent*>::iterator tempAgent =
    neighbourAgents.begin();

while (tempAgent != neighbourAgents.end()) {
    if ((*tempAgent)->getType() == agentType) {
        similarCount++;
    }
    tempAgent++;
}

// 判断同类比例是否达到满意阈值
if (neighbourAgents.empty()) {
    isSatisfied = true;
} else {
    double similarRatio =
        static_cast<double>(similarCount) / neighbourAgents.size();

    isSatisfied = (similarRatio >= threshold);
}
}

void Agent::move(
    repast::SharedDiscreteSpace<
        Agent, repast::StrictBorders,
        repast::SimpleAdder<Agent>
    >* space,
    std::vector<int>& occupied,
    int boardSize) {

    std::vector<int> emptyCells;

    for (int i = 0; i < boardSize * boardSize; ++i) {
        if (occupied[i] == 0) {
            emptyCells.push_back(i);
        }
    }

    if (emptyCells.empty()) {
        return;
    }

    auto gen = repast::Random::instance()
        ->createUniIntGenerator(
            0, static_cast<int>(emptyCells.size()) - 1);

    int target = emptyCells[gen.next()];
    int x = target % boardSize;
    int y = target / boardSize;

    std::vector<int> oldLocation;
    space->getLocation(agentId, oldLocation);

    space->moveTo(agentId, repast::Point<int>(x, y));

    int oldIndex =
        oldLocation[1] * boardSize + oldLocation[0];

    occupied[oldIndex] = 0;
    occupied[target] = 1;
}

AgentPackage::AgentPackage() {
}

AgentPackage::AgentPackage(
    int _id, int _rank, int _type,
    int _currentRank, int _agentType,
    bool _isSatisfied, double _threshold)
    : id(_id),
      rank(_rank),
      type(_type),
      currentRank(_currentRank),
      agentType(_agentType),
      isSatisfied(_isSatisfied),
      threshold(_threshold) {
}