/* Demo_03_Model.cpp */

#include <stdio.h>
#include <vector>
#include <boost/mpi.hpp>
#include "repast_hpc/AgentId.h"
#include "repast_hpc/RepastProcess.h"
#include "repast_hpc/Utilities.h"
#include "repast_hpc/Properties.h"
#include "repast_hpc/initialize_random.h"
#include "repast_hpc/SVDataSetBuilder.h"
#include "repast_hpc/Point.h"
#include "repast_hpc/Random.h"
#include <functional>
#include "Model.h"
#include <stdexcept>

AgentPackageProvider::AgentPackageProvider(
    repast::SharedContext<Agent>* agentPtr)
    : agents(agentPtr) {
}

void AgentPackageProvider::providePackage(
    Agent* agent, std::vector<AgentPackage>& out) {

    repast::AgentId id = agent->getId();

   AgentPackage package(
    id.id(),
    id.startingRank(),
    id.agentType(),
    id.currentRank(),
    agent->getType(),
    agent->getSatisfiedStatus(),
    agent->getThreshold()
);

    out.push_back(package);
}

void AgentPackageProvider::provideContent(
    repast::AgentRequest req,
    std::vector<AgentPackage>& out) {

    std::vector<repast::AgentId> ids = req.requestedAgents();

    for (size_t i = 0; i < ids.size(); i++) {
        providePackage(agents->getAgent(ids[i]), out);
    }
}

AgentPackageReceiver::AgentPackageReceiver(
    repast::SharedContext<Agent>* agentPtr)
    : agents(agentPtr) {
}

Agent* AgentPackageReceiver::createAgent(AgentPackage package) {
    repast::AgentId id(
        package.id,
        package.rank,
        package.type,
        package.currentRank
    );

    return new Agent(
    id,
    package.agentType,
    package.isSatisfied,
    package.threshold
);
}

void AgentPackageReceiver::updateAgent(AgentPackage package) {
    repast::AgentId id(package.id, package.rank, package.type);
    Agent* agent = agents->getAgent(id);

    agent->set(
        package.currentRank,
        package.agentType,
        package.isSatisfied
    );
}

DataSource_AgentSatisfiedTotals::DataSource_AgentSatisfiedTotals(
    repast::SharedContext<Agent>* c)
    : context(c) {
}

int DataSource_AgentSatisfiedTotals::getData() {
    int sum = 0;

    auto iter = context->localBegin();
    auto iterEnd = context->localEnd();

    for (; iter != iterEnd; ++iter) {
        sum += (*iter)->getSatisfiedStatus();
    }

    return sum;
}

SchellingModel::SchellingModel(std::string propsFile, int argc, char** argv, boost::mpi::communicator* comm): context(comm){
	props = new repast::Properties(propsFile, argc, argv, comm);
	stopAt = repast::strToInt(props->getProperty("stop.at"));
	countOfAgents = repast::strToInt(props->getProperty("count.of.agents"));
	boardSize = repast::strToInt(props->getProperty("board.size"));
	initializeRandom(*props, comm);

	repast::Point<double> origin(0, 0);
repast::Point<double> extent(boardSize, boardSize);
repast::GridDimensions gd(origin, extent);

int procX = repast::strToInt(props->getProperty("proc.per.x"));
int procY = repast::strToInt(props->getProperty("proc.per.y"));
int bufferSize = repast::strToInt(props->getProperty("grid.buffer"));

std::vector<int> processDims;
processDims.push_back(procX);
processDims.push_back(procY);

discreteSpace =
    new repast::SharedDiscreteSpace<
        Agent,
        repast::StrictBorders,
        repast::SimpleAdder<Agent>
    >(
        "AgentDiscreteSpace",
        gd,
        processDims,
        bufferSize,
        comm
    );

context.addProjection(discreteSpace);
printf(
    "Rank %d: local origin=(%.1f,%.1f), extent=(%.1f,%.1f)\n",
    repast::RepastProcess::instance()->rank(),
    discreteSpace->dimensions().origin().getX(),
    discreteSpace->dimensions().origin().getY(),
    discreteSpace->dimensions().extents().getX(),
    discreteSpace->dimensions().extents().getY()
);

provider = new AgentPackageProvider(&context);
receiver = new AgentPackageReceiver(&context);

std::string fileOutputName("./outputs/agent_total_data.csv");

repast::SVDataSetBuilder builder(
    fileOutputName.c_str(),
    ",",
    repast::RepastProcess::instance()
        ->getScheduleRunner().schedule()
);

DataSource_AgentSatisfiedTotals* source =
    new DataSource_AgentSatisfiedTotals(&context);

builder.addDataSource(
    repast::createSVDataSource(
        "Total Satisfied",
        source,
        std::plus<int>()
    )
);

agentValues = builder.createDataSet();

}

SchellingModel::~SchellingModel(){
	delete props;
    delete provider;
    delete receiver;
}

void SchellingModel::initAgents() {
    int rank = repast::RepastProcess::instance()->rank();

    int xMin = static_cast<int>(
        discreteSpace->dimensions().origin().getX());
    int yMin = static_cast<int>(
        discreteSpace->dimensions().origin().getY());

    int width = static_cast<int>(
        discreteSpace->dimensions().extents().getX());
    int height = static_cast<int>(
        discreteSpace->dimensions().extents().getY());

    if (countOfAgents < 0 || countOfAgents >= width * height) {
        throw std::runtime_error(
            "Each partition must retain at least one empty cell");
    }

    auto genX = repast::Random::instance()
        ->createUniIntGenerator(xMin, xMin + width - 1);
    auto genY = repast::Random::instance()
        ->createUniIntGenerator(yMin, yMin + height - 1);

    double threshold =
        repast::strToDouble(props->getProperty("threshold"));

    for (int i = 0; i < countOfAgents; ++i) {
        int x, y;
        std::vector<Agent*> occupants;

        do {
            occupants.clear();
            x = genX.next();
            y = genY.next();

            discreteSpace->getObjectsAt(
                repast::Point<int>(x, y), occupants);
        } while (!occupants.empty());

        int type = i % 2;

        repast::AgentId id(i, rank, 0);
        id.currentRank(rank);

        Agent* agent = new Agent(id, type, threshold);
        context.addAgent(agent);
        discreteSpace->moveTo(id, repast::Point<int>(x, y));

        printf("Init: rank %d, type %d, at (%d,%d)\n",
               rank, type, x, y);
    }
}


void SchellingModel::doPerTick() {
    boost::mpi::communicator world;
    int rank = world.rank();

    // 同步归属、空间副本和属性
    auto synchronize = [&]() {
        discreteSpace->balance();

        repast::RepastProcess::instance()->synchronizeAgentStatus<
            Agent, AgentPackage,
            AgentPackageProvider, AgentPackageReceiver
        >(context, *provider, *receiver, *receiver);

        repast::RepastProcess::instance()->synchronizeProjectionInfo<
            Agent, AgentPackage,
            AgentPackageProvider, AgentPackageReceiver
        >(context, *provider, *receiver, *receiver);

        repast::RepastProcess::instance()->synchronizeAgentStates<
            AgentPackage, AgentPackageProvider, AgentPackageReceiver
        >(*provider, *receiver);
    };

    synchronize();

    // 根据本轮开始时的邻居判断满意度
    // 记录原有 Agent，避免迁移后在同一轮重复行动
    std::vector<repast::AgentId> movers;

    for (auto it = context.localBegin();
         it != context.localEnd(); ++it) {
        Agent* agent = &**it;
        agent->updateStatus(&context, discreteSpace);

        if (!agent->getSatisfiedStatus()) {
            movers.push_back(agent->getId());
        }
    }

    // 各进程依次移动，避免同时选中同一个空格
    for (int turn = 0; turn < world.size(); ++turn) {
        std::vector<int> localOccupied(boardSize * boardSize, 0);
        std::vector<int> occupied(boardSize * boardSize, 0);

        // 只统计 LOCAL，避免副本重复计数
        for (auto it = context.localBegin();
             it != context.localEnd(); ++it) {
            Agent* agent = &**it;
            std::vector<int> loc;
            discreteSpace->getLocation(agent->getId(), loc);

            int index = loc[1] * boardSize + loc[0];
            localOccupied[index]++;
        }

        boost::mpi::all_reduce(
            world,
            localOccupied.data(),
            static_cast<int>(localOccupied.size()),
            occupied.data(),
            std::plus<int>()
        );

        if (rank == turn) {
            for (const auto& id : movers) {
                Agent* agent = context.getAgent(id);
                agent->move(discreteSpace, occupied, boardSize);
            }
        }

        // 所有进程共同参与同步
        synchronize();
    }

    std::vector<int> localCounts(boardSize * boardSize, 0);
std::vector<int> globalCounts(boardSize * boardSize, 0);

for (auto it = context.localBegin();
     it != context.localEnd(); ++it) {
    std::vector<int> loc;
    discreteSpace->getLocation((*it)->getId(), loc);
    localCounts[loc[1] * boardSize + loc[0]]++;
}

boost::mpi::all_reduce(
    world,
    localCounts.data(),
    static_cast<int>(localCounts.size()),
    globalCounts.data(),
    std::plus<int>()
);

int total = 0;
bool overlap = false;

for (int count : globalCounts) {
    total += count;
    if (count > 1) {
        overlap = true;
    }
}

if (total != countOfAgents * world.size() || overlap) {
    throw std::runtime_error(
        "Agent count changed or multiple agents occupy one cell");
}

if (rank == 0) {
    printf("Check: agents=%d, overlap=0\n", total);
}

    // 按移动后的空间重新计算满意度，供 CSV 统计
    for (auto it = context.localBegin();
         it != context.localEnd(); ++it) {
        Agent* agent = &**it;
        agent->updateStatus(&context, discreteSpace);
    }

    repast::RepastProcess::instance()->synchronizeAgentStates<
        AgentPackage, AgentPackageProvider, AgentPackageReceiver
    >(*provider, *receiver);

    if (rank == 0) {
        double tick = repast::RepastProcess::instance()
            ->getScheduleRunner().currentTick();

        printf("Tick %.0f completed\n", tick);
    }
}

void SchellingModel::initSchedule(repast::ScheduleRunner& runner){
	runner.scheduleEvent(1, 1, repast::Schedule::FunctorPtr(new repast::MethodFunctor<SchellingModel> (this, &SchellingModel::doPerTick)));
	runner.scheduleStop(stopAt);
    runner.scheduleEvent(
    0, 1,
    repast::Schedule::FunctorPtr(
        new repast::MethodFunctor<repast::DataSet>(
            agentValues, &repast::DataSet::record)
    )
);

runner.scheduleEvent(
    1, 1,
    repast::Schedule::FunctorPtr(
        new repast::MethodFunctor<repast::DataSet>(
            agentValues, &repast::DataSet::write)
    )
);

runner.scheduleEndEvent(
    repast::Schedule::FunctorPtr(
        new repast::MethodFunctor<repast::DataSet>(
            agentValues, &repast::DataSet::write)
    )
);
}

