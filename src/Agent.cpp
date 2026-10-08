#include "Agent.h"
#include "repast_hpc/Moore2DGridQuery.h"
#include "repast_hpc/Point.h"
#include <stdio.h>

Agent::Agent(repast::AgentId id, int type, double threshold): agentId(id), agentType(type), threshold(threshold) {}

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
isSatisfied =
    ((double)similarCount / neighbourAgents.size() >= threshold);
}

void Agent::move(repast::SharedDiscreteSpace<Agent, repast::StrictBorders, repast::SimpleAdder<Agent> >* space){
	// 获取当前坐标
std::vector<int> agentLoc;
space->getLocation(agentId, agentLoc);

// 创建随机坐标生成器
int xMax = (int)space->bounds().extents(0) - 1;

repast::IntUniformGenerator gen =
    repast::Random::instance()->createUniIntGenerator(1, xMax);

std::vector<int> agentNewLoc;

do {
    agentNewLoc.clear();

    int xRand, yRand;
    std::vector<Agent*> agentList;

    // 持续抽取坐标，直到找到空位置
    do {
        agentList.clear();
        xRand = gen.next();
        yRand = gen.next();

        space->getObjectsAt(
            repast::Point<int>(xRand, yRand),
            agentList
        );
    } while (agentList.size() != 0);

    agentNewLoc.push_back(xRand);
    agentNewLoc.push_back(yRand);

} while (!space->bounds().contains(agentNewLoc));

// 移动到新位置
space->moveTo(agentId, agentNewLoc);
}

