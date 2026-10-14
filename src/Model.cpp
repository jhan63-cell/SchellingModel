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
        agent->getSatisfiedStatus()
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

    return new Agent(id, package.agentType, package.isSatisfied);
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

void SchellingModel::initAgents(){

    int rank = repast::RepastProcess::instance()->rank();

double threshold =
    repast::strToDouble(props->getProperty("threshold"));

int x = 0;
int y = 0;

if (rank == 0) {
    x = 1;
    y = 2;
} else if (rank == 1) {
    x = 2;
    y = 3;
} else if (rank == 2) {
    x = 4;
    y = 2;
} else if (rank == 3) {
    x = 5;
    y = 5;
}

repast::AgentId id(0, rank, 0);
id.currentRank(rank);

Agent* agent = new Agent(id, 0, threshold);
context.addAgent(agent);

repast::Point<int> location(x, y);
discreteSpace->moveTo(id, location);

printf("Init: rank %d, agent at (%d,%d)\n", rank, x, y);

/*
    int rank = repast::RepastProcess::instance()->rank();

    repast::IntUniformGenerator gen =
        repast::Random::instance()
            ->createUniIntGenerator(1, boardSize);

    int countType0 = countOfAgents / 2;
    int countType1 = countOfAgents - countType0;

    double threshold =
        repast::strToDouble(props->getProperty("threshold"));

    for (int i = 0; i < countOfAgents; i++) {
        int xRand, yRand;
        std::vector<Agent*> agentList;

        do {
            agentList.clear();
            xRand = gen.next();
            yRand = gen.next();

            discreteSpace->getObjectsAt(
                repast::Point<int>(xRand, yRand),
                agentList
            );
        } while (agentList.size() != 0);

        repast::Point<int> initialLocation(xRand, yRand);

        repast::AgentId id(i, rank, 0);
        id.currentRank(rank);

        int type;
        if (countType0 > 0) {
            type = 0;
            countType0--;
        } else {
            type = 1;
            countType1--;
        }

        Agent* agent = new Agent(id, type, threshold);
        context.addAgent(agent);
        discreteSpace->moveTo(id, initialLocation);
    }
*/
}

void SchellingModel::doPerTick(){

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

int rank = repast::RepastProcess::instance()->rank();

double currentTick =
    repast::RepastProcess::instance()
        ->getScheduleRunner().currentTick();

if (rank == 0) {
    printf("Tick %.1f\n", currentTick);
}

if (rank == 0) {
    printf("Rank 0 - LOCAL\n");

    auto local = context.begin(repast::SharedContext<Agent>::LOCAL);
    auto localEnd = context.end(repast::SharedContext<Agent>::LOCAL);

    for (; local != localEnd; ++local) {
        Agent* agent = &**local;
        std::vector<int> loc;
        discreteSpace->getLocation(agent->getId(), loc);

        printf("Agent at (%d,%d), satisfied=%d\n",
               loc[0], loc[1], agent->getSatisfiedStatus());
    }

    printf("Rank 0 - NON_LOCAL\n");

    auto remote = context.begin(repast::SharedContext<Agent>::NON_LOCAL);
    auto remoteEnd = context.end(repast::SharedContext<Agent>::NON_LOCAL);

    for (; remote != remoteEnd; ++remote) {
        Agent* agent = &**remote;
        std::vector<int> loc;
        discreteSpace->getLocation(agent->getId(), loc);

        printf("Agent at (%d,%d), satisfied=%d\n",
               loc[0], loc[1], agent->getSatisfiedStatus());
    }
}

if (currentTick == 1 && rank == 0) {
    repast::AgentId id(0, 0, 0);

    std::vector<int> newLocation;
    newLocation.push_back(0);
    newLocation.push_back(3);

    discreteSpace->moveTo(id, newLocation);
}

/*
	double avgSatisfied = 0;
std::vector<Agent*> agents;

context.selectAgents(
    repast::SharedContext<Agent>::LOCAL,
    countOfAgents,
    agents
);

std::vector<Agent*>::iterator it = agents.begin();

while (it != agents.end()) {
    (*it)->updateStatus(&context, discreteSpace);
    avgSatisfied += (*it)->getSatisfiedStatus();
    it++;
}

avgSatisfied /= countOfAgents;

double currentTick =
    repast::RepastProcess::instance()
        ->getScheduleRunner().currentTick();

if (repast::RepastProcess::instance()->rank() == 0) {
    printf("Tick: %.1f\tAvg satisfied: %.2f\n",
           currentTick, avgSatisfied);

    if (currentTick == 1 || currentTick == stopAt)
        printToScreen();
}

it = agents.begin();

while (it != agents.end()) {
    if (!(*it)->getSatisfiedStatus()) {
        (*it)->move(discreteSpace);
    }
    it++;
}
*/

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

