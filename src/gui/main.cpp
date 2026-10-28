#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QCloseEvent>
#include <QColor>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QDoubleSpinBox>
#include <QEvent>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QGridLayout>
#include <QGroupBox>
#include <QHash>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProcess>
#include <QPushButton>
#include <QScrollArea>
#include <QShortcut>
#include <QSizePolicy>
#include <QSpinBox>
#include <QSplitter>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QTabWidget>
#include <QTimer>
#include <QToolButton>
#include <QUrl>
#include <QVBoxLayout>

namespace {

QString textOf(const QJsonValue& value, int digits = 3) {
    if (value.isNull() || value.isUndefined()) return QStringLiteral("—");
    if (value.isDouble()) return QString::number(value.toDouble(), 'f', digits);
    return value.toVariant().toString();
}

QJsonObject objectFromFile(const QString& path) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return {};
    QJsonParseError error{};
    const auto document = QJsonDocument::fromJson(file.readAll(), &error);
    return error.error == QJsonParseError::NoError && document.isObject() ? document.object() : QJsonObject{};
}

QLabel* caption(const QString& text) {
    auto* label = new QLabel(text);
    label->setObjectName(QStringLiteral("caption"));
    return label;
}

class InspectionWindow final : public QMainWindow {
public:
    InspectionWindow() : root_(QDir(QCoreApplication::applicationDirPath()).absoluteFilePath(QStringLiteral(".."))) {
        setWindowTitle(QStringLiteral("Edge Visual Inspection · 检测工作台"));
        resize(1380, 960);
        setMinimumSize(900, 650);
        buildUi();
        connect(&pollTimer_, &QTimer::timeout, this, [this] { pollEvents(); });
        connect(&process_, &QProcess::readyRead, this, [this] {
            appendLog(QString::fromLocal8Bit(process_.readAll()));
        });
        connect(&process_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError) {
            status_->setText(QStringLiteral("启动或运行失败：") + process_.errorString());
            appendLog(process_.errorString());
            start_->setEnabled(true);
            stop_->setEnabled(false);
        });
        connect(&process_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this,
                [this](int code, QProcess::ExitStatus state) {
            pollEvents();
            pollTimer_.stop();
            loadSummary();
            start_->setEnabled(true);
            stop_->setEnabled(false);
            output_->clear();
            status_->setText(aborted_ ? QStringLiteral("用户中止；该次运行不计入正式验收") : state == QProcess::NormalExit
                ? QStringLiteral("进程已结束 · 退出码 %1；请核对 summary 与逐帧事件").arg(code)
                : QStringLiteral("进程异常结束；本次结果不能用于正式验收"));
            appendLog(QStringLiteral("[GUI] 检测进程退出码 %1").arg(code));
        });
        configureBackend();
    }

    void startSmokeRun(bool capturePreview = false) {
        manifest_->setText(QStringLiteral("tests/fixtures/conv/model_manifest.json"));
        source_->setText(QStringLiteral("tests/fixtures/conv/replay.json"));
        updateModelInfo();
        output_->setText(QStringLiteral("runs/gui-smoke-") + QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd-HHmmss-zzz")));
        warmup_->setValue(1);
        deadline_->setValue(1000);
        connect(&process_, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
                qApp, [this, capturePreview](int code, QProcess::ExitStatus state) {
            if (!capturePreview) { qApp->exit(state == QProcess::NormalExit ? code : 4); return; }
            if (events_->rowCount() > 0) events_->selectRow(0);
            QTimer::singleShot(250, this, [this, code, state] {
                resize(1380, 960);
                const auto path = QDir(QCoreApplication::applicationDirPath()).absoluteFilePath(QStringLiteral("../reports/gui-run-preview.png"));
                const bool saved = grab().save(path);
                qApp->exit(saved && state == QProcess::NormalExit ? code : 4);
            });
        });
        connect(&process_, &QProcess::errorOccurred, qApp, [](QProcess::ProcessError) { qApp->exit(4); });
        QTimer::singleShot(0, this, [this] { startRun(); });
        QTimer::singleShot(30000, qApp, [] { qApp->exit(4); });
    }

protected:
    bool eventFilter(QObject* watched, QEvent* event) override {
        if ((watched == originalPreview_ || watched == preview_) && event->type() == QEvent::Resize) {
            updatePreviewImages();
        }
        return QMainWindow::eventFilter(watched, event);
    }

    void closeEvent(QCloseEvent* event) override {
        if (process_.state() != QProcess::NotRunning) {
            const auto reply = QMessageBox::question(this, QStringLiteral("检测仍在运行"),
                QStringLiteral("关闭界面会中止检测进程，当前回放可能没有完整 summary。确定退出吗？"));
            if (reply != QMessageBox::Yes) { event->ignore(); return; }
            process_.kill();
            process_.waitForFinished(3000);
        }
        event->accept();
    }

private:
    QString absolute(const QString& path) const {
        return QDir::isAbsolutePath(path) ? QDir::cleanPath(path) : root_.absoluteFilePath(path);
    }

    void buildUi() {
        setStyleSheet(QStringLiteral(R"(
            QMainWindow, QWidget { background:#101820; color:#e5edf3; font-family:'Microsoft YaHei UI'; font-size:16px; }
            QTabWidget::pane { border:1px solid #304354; border-radius:8px; background:#14202b; }
            QTabBar::tab { background:#1b2a37; color:#c2d0d9; padding:9px 20px; margin-right:5px; border-radius:6px; }
            QTabBar::tab:selected { background:#147e8a; color:white; }
            QGroupBox { border:1px solid #304354; border-radius:10px; margin-top:16px; padding:10px; background:#16232f; font-weight:600; }
            QGroupBox::title { subcontrol-origin:margin; left:14px; padding:0 6px; color:#b4dce2; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTableWidget {
                background:#0e1822; color:#e5edf3; border:1px solid #3a5161; border-radius:5px; padding:5px;
            }
            QComboBox QAbstractItemView { background:#172633; color:#e5edf3; selection-background-color:#147e8a; }
            QPushButton { background:#273c4c; color:white; border:1px solid #466074; border-radius:6px; padding:8px 14px; }
            QPushButton:hover { background:#365365; }
            QPushButton:focus, QToolButton:focus, QLineEdit:focus, QComboBox:focus,
            QSpinBox:focus, QDoubleSpinBox:focus, QCheckBox:focus, QTableWidget:focus,
            QPlainTextEdit:focus, QTabBar::tab:focus { border:2px solid #79d9e3; }
            QPushButton:disabled { background:#22303b; color:#718592; }
            QPushButton#primary { background:#087e8b; border-color:#19a4b2; font-weight:700; }
            QPushButton#danger { background:#804048; border-color:#a55a61; }
            QToolButton { color:#cceaf0; background:#203543; border:1px solid #466074; border-radius:6px; padding:7px 12px; }
            QToolButton:hover { background:#365365; }
            QLabel#heading { font-size:24px; font-weight:700; color:#f5f9fc; }
            QLabel#subheading { color:#a9bfcc; font-size:14px; }
            QLabel#caption { color:#c5d4dc; }
            QLabel#step { color:#9ae3e8; font-size:17px; font-weight:600; }
            QLabel#imageTitle { color:#c8dce6; font-weight:600; }
            QLabel#imageSurface { background:#0c1720; border:1px solid #385063; border-radius:6px; }
            QLabel#status { background:#19313d; border:1px solid #326477; border-radius:6px; padding:8px; }
            QHeaderView::section { background:#243847; color:#dce9ef; border:0; padding:7px; }
            QTableWidget { gridline-color:#294050; alternate-background-color:#182936; selection-background-color:#176b78; selection-color:#ffffff; }
            QScrollBar:vertical { background:#14202b; width:11px; }
        )"));
        auto* central = new QWidget;
        auto* outer = new QVBoxLayout(central);
        outer->setContentsMargins(16, 12, 16, 12);
        auto* header = new QHBoxLayout;
        auto* titles = new QVBoxLayout;
        auto* title = new QLabel(QStringLiteral("工业视觉质检 · 检测工作台"));
        title->setObjectName(QStringLiteral("heading"));
        titles->addWidget(title);
        auto* sub = new QLabel(QStringLiteral("选择模型和图片，开始检测，再查看逐帧事件与图像"));
        sub->setObjectName(QStringLiteral("subheading"));
        titles->addWidget(sub);
        header->addLayout(titles, 1);
        auto* rootLabel = caption(QStringLiteral("工作区：") + QDir::toNativeSeparators(root_.absolutePath()));
        rootLabel->setToolTip(rootLabel->text());
        rootLabel->setMinimumWidth(0);
        header->addWidget(rootLabel, 0, Qt::AlignRight);
        outer->addLayout(header);

        auto* tabs = new QTabWidget;
        tabs->addTab(buildDetectionTab(), QStringLiteral("检测与回放"));
        tabs->addTab(buildProjectTab(), QStringLiteral("实验与文档"));
        outer->addWidget(tabs, 1);
        setCentralWidget(central);
    }

    QWidget* buildDetectionTab() {
        auto* page = new QWidget;
        auto* layout = new QVBoxLayout(page);
        layout->setContentsMargins(12, 8, 12, 10);
        layout->setSpacing(7);
        auto* config = new QGroupBox(QStringLiteral("1  选择模型与图片"));
        auto* grid = new QGridLayout(config);
        grid->setHorizontalSpacing(10);
        grid->setVerticalSpacing(7);
        grid->setColumnStretch(1, 1);
        grid->setColumnStretch(3, 1);

        backend_ = new QComboBox;
        backend_->addItem(QStringLiteral("ORT CUDA · FP32 主线"), QStringLiteral("ort-cuda"));
        backend_->addItem(QStringLiteral("TensorRT INT8 · 验证路线"), QStringLiteral("trt-int8"));
        backend_->addItem(QStringLiteral("ORT CPU · 诊断用"), QStringLiteral("ort-cpu"));
        backend_->addItem(QStringLiteral("TensorRT FP32 · 仅计时对照"), QStringLiteral("trt-fp32"));
        grid->addWidget(caption(QStringLiteral("推理后端")), 0, 0);
        grid->addWidget(backend_, 0, 1);
        task_ = new QComboBox;
        task_->addItem(QStringLiteral("逐帧检测 / 回放"), QStringLiteral("run"));
        task_->addItem(QStringLiteral("基准测试"), QStringLiteral("benchmark"));
        grid->addWidget(caption(QStringLiteral("运行方式")), 0, 2);
        grid->addWidget(task_, 0, 3);

        manifest_ = new QLineEdit;
        manifest_->setPlaceholderText(QStringLiteral("选择 model_manifest.json"));
        addPath(grid, 1, 0, QStringLiteral("模型清单"), manifest_, false);
        source_ = new QLineEdit(QStringLiteral("data/replay.json"));
        source_->setPlaceholderText(QStringLiteral("图片目录或回放 JSON"));
        auto* sourceBox = new QWidget;
        auto* sourceLayout = new QHBoxLayout(sourceBox);
        sourceLayout->setContentsMargins(0, 0, 0, 0);
        sourceLayout->addWidget(source_, 1);
        auto* imageDir = new QPushButton(QStringLiteral("选择图片目录"));
        auto* replay = new QPushButton(QStringLiteral("选择 JSON"));
        sourceLayout->addWidget(imageDir);
        sourceLayout->addWidget(replay);
        connect(imageDir, &QPushButton::clicked, this, [this] {
            const auto path = QFileDialog::getExistingDirectory(this, QStringLiteral("选择图片目录"), root_.absolutePath());
            if (!path.isEmpty()) source_->setText(path);
        });
        connect(replay, &QPushButton::clicked, this, [this] {
            const auto path = QFileDialog::getOpenFileName(this, QStringLiteral("选择回放 JSON"), root_.absolutePath(), QStringLiteral("JSON (*.json)"));
            if (!path.isEmpty()) source_->setText(path);
        });
        auto* sourceLabel = caption(QStringLiteral("图片来源"));
        sourceLabel->setBuddy(source_);
        grid->addWidget(sourceLabel, 2, 0);
        grid->addWidget(sourceBox, 2, 1, 1, 3);

        modelInfo_ = caption(QStringLiteral("正在读取模型清单…"));
        modelInfo_->setWordWrap(true);
        grid->addWidget(modelInfo_, 3, 0, 1, 4);
        layout->addWidget(config);

        auto* advancedToggle = new QToolButton;
        advancedToggle->setText(QStringLiteral("更多运行设置"));
        advancedToggle->setCheckable(true);
        advancedToggle->setArrowType(Qt::RightArrow);
        advancedToggle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        advancedToggle->setAccessibleName(QStringLiteral("展开或收起更多运行设置"));
        layout->addWidget(advancedToggle, 0, Qt::AlignLeft);
        auto* advanced = new QGroupBox(QStringLiteral("输出、调度与基准设置"));
        advanced->setVisible(false);
        auto* advancedGrid = new QGridLayout(advanced);
        advancedGrid->setColumnStretch(1, 1);
        advancedGrid->setColumnStretch(3, 1);
        connect(advancedToggle, &QToolButton::toggled, advanced, &QWidget::setVisible);
        connect(advancedToggle, &QToolButton::toggled, this, [advancedToggle](bool expanded) {
            advancedToggle->setArrowType(expanded ? Qt::DownArrow : Qt::RightArrow);
            advancedToggle->setText(expanded ? QStringLiteral("收起运行设置") : QStringLiteral("更多运行设置"));
        });

        output_ = new QLineEdit;
        auto* outputBox = new QWidget;
        auto* outputLayout = new QHBoxLayout(outputBox);
        outputLayout->setContentsMargins(0, 0, 0, 0);
        outputLayout->addWidget(output_, 1);
        auto* outputBrowse = new QPushButton(QStringLiteral("选择目录"));
        outputLayout->addWidget(outputBrowse);
        connect(outputBrowse, &QPushButton::clicked, this, [this] {
            const auto path = QFileDialog::getExistingDirectory(this, QStringLiteral("选择输出父目录"), root_.absoluteFilePath(QStringLiteral("runs")));
            if (!path.isEmpty()) output_->setText(QDir(path).filePath(QStringLiteral("gui-") + QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd-HHmmss-zzz"))));
        });
        auto* outputLabel = caption(QStringLiteral("输出目录"));
        outputLabel->setBuddy(output_);
        advancedGrid->addWidget(outputLabel, 0, 0);
        advancedGrid->addWidget(outputBox, 0, 1, 1, 3);

        engine_ = new QLineEdit;
        evidence_ = new QLineEdit;
        addPath(advancedGrid, 1, 0, QStringLiteral("TRT 引擎"), engine_, false);
        addPath(advancedGrid, 2, 0, QStringLiteral("引擎证据"), evidence_, false);

        fps_ = new QDoubleSpinBox;
        fps_->setRange(0.1, 10000); fps_->setValue(20); fps_->setDecimals(1);
        deadline_ = new QSpinBox;
        deadline_->setRange(1, 600000); deadline_->setValue(50); deadline_->setSuffix(QStringLiteral(" ms"));
        queue_ = new QSpinBox;
        queue_->setRange(1, 10000); queue_->setValue(4);
        warmup_ = new QSpinBox;
        warmup_->setRange(0, 100000); warmup_->setValue(100);
        advancedGrid->addWidget(caption(QStringLiteral("帧率 FPS")), 3, 0); advancedGrid->addWidget(fps_, 3, 1);
        advancedGrid->addWidget(caption(QStringLiteral("截止期")), 3, 2); advancedGrid->addWidget(deadline_, 3, 3);
        advancedGrid->addWidget(caption(QStringLiteral("队列容量")), 4, 0); advancedGrid->addWidget(queue_, 4, 1);
        advancedGrid->addWidget(caption(QStringLiteral("预热次数")), 4, 2); advancedGrid->addWidget(warmup_, 4, 3);

        loop_ = new QCheckBox(QStringLiteral("循环回放"));
        duration_ = new QSpinBox;
        duration_->setRange(1, 86400); duration_->setValue(60); duration_->setSuffix(QStringLiteral(" 秒"));
        cudaPost_ = new QCheckBox(QStringLiteral("CUDA 后处理")); cudaPost_->setChecked(true);
        benchmarkMode_ = new QComboBox;
        benchmarkMode_->addItems({QStringLiteral("pipeline"), QStringLiteral("model"), QStringLiteral("replay")});
        iterations_ = new QSpinBox;
        iterations_->setRange(1, 1000000); iterations_->setValue(1000);
        advancedGrid->addWidget(loop_, 5, 0); advancedGrid->addWidget(duration_, 5, 1);
        advancedGrid->addWidget(cudaPost_, 5, 2);
        advancedGrid->addWidget(caption(QStringLiteral("基准模式")), 6, 0); advancedGrid->addWidget(benchmarkMode_, 6, 1);
        advancedGrid->addWidget(caption(QStringLiteral("基准迭代")), 6, 2); advancedGrid->addWidget(iterations_, 6, 3);
        layout->addWidget(advanced);

        auto* stepTwo = new QLabel(QStringLiteral("2  开始检测"));
        stepTwo->setObjectName(QStringLiteral("step"));
        layout->addWidget(stepTwo);
        auto* controls = new QHBoxLayout;
        start_ = new QPushButton(QStringLiteral("▶ 开始检测")); start_->setObjectName(QStringLiteral("primary"));
        start_->setMinimumHeight(40);
        start_->setToolTip(QStringLiteral("快捷键：Ctrl+Enter"));
        stop_ = new QPushButton(QStringLiteral("■ 中止检测")); stop_->setObjectName(QStringLiteral("danger")); stop_->setEnabled(false);
        openOutput_ = new QPushButton(QStringLiteral("打开输出目录"));
        openOutput_->setEnabled(false);
        status_ = new QLabel(QStringLiteral("就绪 · 选择模型与图片后开始检测"));
        status_->setObjectName(QStringLiteral("status"));
        status_->setWordWrap(true);
        controls->addWidget(start_); controls->addWidget(stop_); controls->addWidget(openOutput_);
        controls->addWidget(status_, 1);
        layout->addLayout(controls);

        auto* stepThree = new QLabel(QStringLiteral("3  查看事件、原图和热图"));
        stepThree->setObjectName(QStringLiteral("step"));
        layout->addWidget(stepThree);

        auto* splitter = new QSplitter(Qt::Horizontal);
        splitter->setMinimumHeight(280);
        auto* eventBox = new QGroupBox(QStringLiteral("逐帧事件 · 最近 500 条"));
        auto* eventLayout = new QVBoxLayout(eventBox);
        eventCount_ = caption(QStringLiteral("事件 0 条")); eventLayout->addWidget(eventCount_);
        events_ = new QTableWidget(0, 7);
        events_->setHorizontalHeaderLabels({QStringLiteral("帧"), QStringLiteral("来源 ID"), QStringLiteral("状态"), QStringLiteral("决策"), QStringLiteral("分数"), QStringLiteral("决策 ms"), QStringLiteral("GPU ms")});
        events_->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
        events_->horizontalHeader()->setSectionResizeMode(1, QHeaderView::Stretch);
        events_->horizontalHeader()->setMinimumSectionSize(65);
        events_->setSelectionBehavior(QAbstractItemView::SelectRows);
        events_->setSelectionMode(QAbstractItemView::SingleSelection);
        events_->setTabKeyNavigation(false);
        events_->setEditTriggers(QAbstractItemView::NoEditTriggers);
        events_->setAlternatingRowColors(true);
        events_->verticalHeader()->setVisible(false);
        eventLayout->addWidget(events_);
        splitter->addWidget(eventBox);
        auto* right = new QWidget;
        auto* rightLayout = new QVBoxLayout(right);
        auto* previewBox = new QGroupBox(QStringLiteral("原图 / 复检热图"));
        auto* previewLayout = new QHBoxLayout(previewBox);
        previewLayout->setSpacing(8);
        auto* originalColumn = new QVBoxLayout;
        auto* originalTitle = new QLabel(QStringLiteral("原图"));
        originalTitle->setObjectName(QStringLiteral("imageTitle"));
        originalColumn->addWidget(originalTitle);
        originalPreview_ = new QLabel(QStringLiteral("选择事件查看原图"));
        originalPreview_->setAlignment(Qt::AlignCenter);
        originalPreview_->setObjectName(QStringLiteral("imageSurface"));
        originalPreview_->setMinimumSize(140, 160);
        originalPreview_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Expanding);
        originalPreview_->setWordWrap(true);
        originalPreview_->installEventFilter(this);
        originalColumn->addWidget(originalPreview_, 1);
        auto* heatColumn = new QVBoxLayout;
        auto* heatTitle = new QLabel(QStringLiteral("复检热图"));
        heatTitle->setObjectName(QStringLiteral("imageTitle"));
        heatColumn->addWidget(heatTitle);
        preview_ = new QLabel(QStringLiteral("选择 REVIEW 事件查看热图"));
        preview_->setAlignment(Qt::AlignCenter);
        preview_->setObjectName(QStringLiteral("imageSurface"));
        preview_->setMinimumSize(140, 160);
        preview_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Expanding);
        preview_->setWordWrap(true);
        preview_->installEventFilter(this);
        heatColumn->addWidget(preview_, 1);
        previewLayout->addLayout(originalColumn, 1);
        previewLayout->addLayout(heatColumn, 1);
        rightLayout->addWidget(previewBox, 3);
        auto* summaryBox = new QGroupBox(QStringLiteral("运行摘要"));
        auto* summaryLayout = new QVBoxLayout(summaryBox);
        summary_ = new QLabel(QStringLiteral("运行结束后读取 summary.json；状态和质量以报告为准。"));
        summary_->setWordWrap(true);
        summaryLayout->addWidget(summary_);
        rightLayout->addWidget(summaryBox, 1);
        splitter->addWidget(right);
        splitter->setStretchFactor(0, 3); splitter->setStretchFactor(1, 2);
        splitter->setChildrenCollapsible(false);
        layout->addWidget(splitter, 1);
        auto* logToggle = new QToolButton;
        logToggle->setText(QStringLiteral("查看进程输出"));
        logToggle->setCheckable(true);
        logToggle->setArrowType(Qt::RightArrow);
        logToggle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        layout->addWidget(logToggle, 0, Qt::AlignLeft);
        auto* logBox = new QGroupBox(QStringLiteral("进程输出"));
        auto* logLayout = new QVBoxLayout(logBox);
        log_ = new QPlainTextEdit; log_->setReadOnly(true); log_->setMaximumBlockCount(1000);
        logLayout->addWidget(log_);
        logBox->setVisible(false);
        logBox->setMaximumHeight(150);
        connect(logToggle, &QToolButton::toggled, logBox, &QWidget::setVisible);
        connect(logToggle, &QToolButton::toggled, this, [logToggle](bool expanded) {
            logToggle->setArrowType(expanded ? Qt::DownArrow : Qt::RightArrow);
            logToggle->setText(expanded ? QStringLiteral("收起进程输出") : QStringLiteral("查看进程输出"));
        });
        layout->addWidget(logBox);

        connect(backend_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this] { configureBackend(); });
        connect(task_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this] { configureTask(); });
        connect(manifest_, &QLineEdit::editingFinished, this, [this] { updateModelInfo(); });
        connect(start_, &QPushButton::clicked, this, [this] { startRun(); });
        connect(stop_, &QPushButton::clicked, this, [this] { stopRun(); });
        connect(openOutput_, &QPushButton::clicked, this, [this] {
            const auto path = activeOutput_.isEmpty() ? absolute(output_->text()) : activeOutput_;
            if (QFileInfo::exists(path)) QDesktopServices::openUrl(QUrl::fromLocalFile(path));
        });
        connect(events_, &QTableWidget::itemSelectionChanged, this, [this] { showSelectedPreview(); });
        auto* startShortcut = new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_Return), this);
        connect(startShortcut, &QShortcut::activated, this, [this] { if (start_->isEnabled()) startRun(); });
        configureTask();
        auto* scroll = new QScrollArea;
        scroll->setWidgetResizable(true);
        scroll->setFrameShape(QFrame::NoFrame);
        scroll->setWidget(page);
        return scroll;
    }

    QWidget* buildProjectTab() {
        auto* page = new QWidget;
        auto* layout = new QVBoxLayout(page);
        layout->setContentsMargins(20, 22, 20, 22);
        auto* intro = new QLabel(QStringLiteral("已有 metal_nut 模型与实验记录保持只读。训练、量化与最终测试应在独立工作副本中执行，以免改写冻结证据。"));
        intro->setWordWrap(true);
        layout->addWidget(intro);
        auto* docs = new QGroupBox(QStringLiteral("文档与证据"));
        auto* docLayout = new QGridLayout(docs);
        const struct { const char* label; const char* path; } links[] = {
            {"使用手册", "docs/USER_GUIDE.md"}, {"项目介绍", "docs/PROJECT_OVERVIEW.md"},
            {"完整实验复现", "docs/REPRODUCIBILITY.md"}, {"执行状态", "docs/STATUS.md"},
            {"独立验收", "docs/ACCEPTANCE.md"}, {"研究报告", "docs/REPORT.md"},
            {"最终质量 JSON", "reports/final_quality.json"}, {"INT8 执行证据", "reports/trt_verified_selective_2.json"}
        };
        for (int i = 0; i < 8; ++i) {
            auto* button = new QPushButton(QString::fromUtf8(links[i].label));
            const auto path = QString::fromUtf8(links[i].path);
            connect(button, &QPushButton::clicked, this, [this, path] { QDesktopServices::openUrl(QUrl::fromLocalFile(absolute(path))); });
            docLayout->addWidget(button, i / 4, i % 4);
        }
        layout->addWidget(docs);
        auto* workflow = new QGroupBox(QStringLiteral("离线流程命令 · 阅读与复制"));
        auto* workflowLayout = new QVBoxLayout(workflow);
        auto* note = caption(QStringLiteral("Python 仅用于离线训练、导出和评测；在线检测由 C++ inspection.exe 完成。以下命令不会从本界面自动执行。"));
        note->setWordWrap(true); workflowLayout->addWidget(note);
        auto* commands = new QPlainTextEdit;
        commands->setReadOnly(true);
        commands->setPlainText(QStringLiteral(
            ".\\.venv-tools\\Scripts\\python.exe tools/prepare_data.py --config configs/metal_nut.yaml\n"
            ".\\.venv-train\\Scripts\\python.exe tools/train.py --config configs/metal_nut.yaml\n"
            ".\\.venv-train\\Scripts\\python.exe tools/export_onnx.py --config configs/metal_nut.yaml\n"
            ".\\.venv-train\\Scripts\\python.exe tools/verify_alignment.py --config configs/metal_nut.yaml --split validation --cpp\n"
            ".\\.venv-tools\\Scripts\\python.exe tools/evaluate.py --config configs/metal_nut.yaml --split validation\n"
            ".\\.venv-tools\\Scripts\\python.exe tools/quantize.py --config configs/metal_nut.yaml --phase base\n"
            ".\\.venv-tools\\Scripts\\python.exe tools/pipeline.py --stage P8"));
        workflowLayout->addWidget(commands);
        auto* copy = new QPushButton(QStringLiteral("复制所选命令／全文"));
        connect(copy, &QPushButton::clicked, this, [commands] {
            QApplication::clipboard()->setText(commands->textCursor().hasSelection() ? commands->textCursor().selectedText().replace(QChar(0x2029), QChar('\n')) : commands->toPlainText());
        });
        workflowLayout->addWidget(copy);
        layout->addWidget(workflow, 1);
        return page;
    }

    void addPath(QGridLayout* grid, int row, int col, const QString& label, QLineEdit* edit, bool directory) {
        auto* box = new QWidget;
        auto* line = new QHBoxLayout(box); line->setContentsMargins(0, 0, 0, 0);
        line->addWidget(edit, 1);
        auto* browse = new QPushButton(QStringLiteral("浏览"));
        line->addWidget(browse);
        connect(browse, &QPushButton::clicked, this, [this, edit, directory] {
            const auto path = directory ? QFileDialog::getExistingDirectory(this, QStringLiteral("选择目录"), root_.absolutePath())
                : QFileDialog::getOpenFileName(this, QStringLiteral("选择文件"), root_.absolutePath());
            if (!path.isEmpty()) { edit->setText(path); updateModelInfo(); }
        });
        grid->addWidget(caption(label), row, col);
        grid->addWidget(box, row, col + 1, 1, 3);
    }

    void configureBackend() {
        const auto backend = backend_->currentData().toString();
        const bool trt = backend.startsWith(QStringLiteral("trt-"));
        engine_->setEnabled(trt); evidence_->setEnabled(trt);
        if (backend == QStringLiteral("trt-int8")) {
            manifest_->setText(QStringLiteral("artifacts/metal_nut/int8_selective_2_trt_compat/model_manifest.json"));
            engine_->setText(QStringLiteral("runs/trt-detailed-int8_selective_2_trt_compat/model.engine"));
            evidence_->setText(QStringLiteral("reports/trt_verified_selective_2.json"));
        } else {
            manifest_->setText(QStringLiteral("artifacts/metal_nut/fp32/model_manifest.json"));
            if (backend == QStringLiteral("trt-fp32")) {
                engine_->setText(QStringLiteral("runs/trt-detailed-fp32/model.engine"));
                evidence_->setText(QStringLiteral("runs/trt-detailed-fp32/evidence.json"));
            } else { engine_->clear(); evidence_->clear(); }
        }
        updateModelInfo();
    }

    void configureTask() {
        const bool bench = task_->currentData().toString() == QStringLiteral("benchmark");
        benchmarkMode_->setEnabled(bench); iterations_->setEnabled(bench);
        loop_->setEnabled(!bench); duration_->setEnabled(!bench);
    }

    void updateModelInfo() {
        const auto manifest = objectFromFile(absolute(manifest_->text()));
        if (manifest.isEmpty()) { modelInfo_->setText(QStringLiteral("模型清单不可读取。请选择现有 model_manifest.json。")); return; }
        const auto threshold = manifest.value(QStringLiteral("image_threshold"));
        modelInfo_->setText(QStringLiteral("模型 %1  ·  类别 %2  ·  精度 %3  ·  状态 %4  ·  图像阈值 %5")
            .arg(manifest.value(QStringLiteral("model_id")).toString(), manifest.value(QStringLiteral("category")).toString(),
                manifest.value(QStringLiteral("requested_precision")).toString(), manifest.value(QStringLiteral("status")).toString(), textOf(threshold, 6)));
    }

    void startRun() {
        if (process_.state() != QProcess::NotRunning) return;
        const auto exe = absolute(QStringLiteral("build/inspection.exe"));
        const auto manifest = absolute(manifest_->text().trimmed());
        const auto source = absolute(source_->text().trimmed());
        const auto backend = backend_->currentData().toString();
        const bool trt = backend.startsWith(QStringLiteral("trt-"));
        if (!QFileInfo(exe).isFile() || !QFileInfo(manifest).isFile() || !QFileInfo::exists(source)) {
            QMessageBox::warning(this, QStringLiteral("输入缺失"), QStringLiteral("请检查 inspection.exe、模型清单和图片来源是否存在。")); return;
        }
        if (trt && (!QFileInfo(absolute(engine_->text())).isFile() || !QFileInfo(absolute(evidence_->text())).isFile())) {
            QMessageBox::warning(this, QStringLiteral("引擎缺失"), QStringLiteral("TensorRT 后端需要真实 engine 和匹配的证据 JSON。")); return;
        }
        QString selectedOutput = output_->text().trimmed();
        if (selectedOutput.isEmpty()) {
            selectedOutput = QStringLiteral("runs/gui-") + QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd-HHmmss-zzz"));
            output_->setText(selectedOutput);
        }
        const auto out = absolute(selectedOutput);
        if (QFileInfo::exists(QDir(out).filePath(QStringLiteral("events.jsonl")))) {
            QMessageBox::warning(this, QStringLiteral("输出已存在"), QStringLiteral("该目录已有 events.jsonl。请选择新目录，保留原始证据。")); return;
        }
        if (!QDir().mkpath(out)) { QMessageBox::warning(this, QStringLiteral("输出不可写"), out); return; }
        activeOutput_ = out; eventOffset_ = 0; totalEvents_ = 0; aborted_ = false; sourcePaths_.clear(); events_->setRowCount(0);
        openOutput_->setEnabled(true);
        originalImage_ = QPixmap(); heatImage_ = QPixmap();
        originalPreview_->setPixmap(QPixmap()); originalPreview_->setText(QStringLiteral("选择事件查看原图"));
        preview_->setPixmap(QPixmap()); preview_->setText(QStringLiteral("选择 REVIEW 事件查看热图"));
        summary_->setText(QStringLiteral("运行中；summary.json 将在正常结束时生成。"));
        eventCount_->setText(QStringLiteral("事件 0 条")); log_->clear();

        QStringList args{task_->currentData().toString(), QStringLiteral("--manifest"), manifest,
            QStringLiteral("--source"), source, QStringLiteral("--backend"), backend,
            QStringLiteral("--output"), out, QStringLiteral("--postprocess"), cudaPost_->isChecked() ? QStringLiteral("cuda") : QStringLiteral("cpu"),
            QStringLiteral("--fps"), QString::number(fps_->value()), QStringLiteral("--deadline-ms"), QString::number(deadline_->value()),
            QStringLiteral("--queue-capacity"), QString::number(queue_->value()), QStringLiteral("--warmup"), QString::number(warmup_->value())};
        if (trt) args << QStringLiteral("--engine") << absolute(engine_->text())
                      << QStringLiteral("--engine-evidence") << absolute(evidence_->text());
        if (task_->currentData().toString() == QStringLiteral("benchmark")) {
            args << QStringLiteral("--mode") << benchmarkMode_->currentText()
                 << QStringLiteral("--iterations") << QString::number(iterations_->value());
        } else if (loop_->isChecked()) {
            args << QStringLiteral("--loop") << QStringLiteral("--duration-seconds") << QString::number(duration_->value());
        }
        process_.setWorkingDirectory(root_.absolutePath());
        process_.setProcessChannelMode(QProcess::MergedChannels);
        process_.setProgram(exe);
        process_.setArguments(args);
        appendLog(QStringLiteral("[GUI] 启动 C++ 检测：") + exe + QStringLiteral(" ") + args.join(QStringLiteral(" ")));
        process_.start();
        start_->setEnabled(false); stop_->setEnabled(true);
        status_->setText(QStringLiteral("运行中 · 请查看逐帧事件"));
        pollTimer_.start(350);
    }

    void stopRun() {
        if (process_.state() == QProcess::NotRunning) return;
        aborted_ = true;
        status_->setText(QStringLiteral("正在中止；当前回放可能没有完整 summary，不能用于验收"));
        process_.terminate();
        QTimer::singleShot(3000, this, [this] { if (process_.state() != QProcess::NotRunning) process_.kill(); });
    }

    void pollEvents() {
        if (activeOutput_.isEmpty()) return;
        if (sourcePaths_.isEmpty()) {
            const auto snapshot = objectFromFile(QDir(activeOutput_).filePath(QStringLiteral("snapshot.json")));
            for (const auto& value : snapshot.value(QStringLiteral("source")).toArray()) {
                const auto row = value.toObject();
                sourcePaths_.insert(row.value(QStringLiteral("source_image_id")).toString(), row.value(QStringLiteral("path")).toString());
            }
        }
        QFile file(QDir(activeOutput_).filePath(QStringLiteral("events.jsonl")));
        if (!file.open(QIODevice::ReadOnly) || !file.seek(eventOffset_)) return;
        while (!file.atEnd()) {
            const qint64 before = file.pos();
            const QByteArray line = file.readLine();
            if (!line.endsWith('\n')) { eventOffset_ = before; break; }
            eventOffset_ = file.pos();
            const auto document = QJsonDocument::fromJson(line);
            if (!document.isObject()) { appendLog(QStringLiteral("[GUI] 跳过无法解析的事件行")); continue; }
            addEvent(document.object());
        }
        eventCount_->setText(QStringLiteral("事件 %1 条 · 表格展示最近 500 条").arg(totalEvents_));
    }

    void addEvent(const QJsonObject& event) {
        ++totalEvents_;
        if (events_->rowCount() == 500) events_->removeRow(0);
        const int row = events_->rowCount(); events_->insertRow(row);
        const auto id = event.value(QStringLiteral("source_image_id")).toString();
        const QStringList cells{
            textOf(event.value(QStringLiteral("frame_id")), 0), id.left(10),
            event.value(QStringLiteral("status")).toString(), event.value(QStringLiteral("decision")).toString(),
            textOf(event.value(QStringLiteral("score")), 5), textOf(event.value(QStringLiteral("decision_latency_ms"))),
            textOf(event.value(QStringLiteral("model_gpu_ms")))};
        for (int col = 0; col < cells.size(); ++col) {
            auto* item = new QTableWidgetItem(cells[col]);
            item->setToolTip(col == 1 ? id : cells[col]);
            if (col == 0) item->setData(Qt::UserRole, event.value(QStringLiteral("frame_id")).toInt());
            if (col == 3 && cells[col] == QStringLiteral("REVIEW")) item->setForeground(QColor(QStringLiteral("#ffb55e")));
            events_->setItem(row, col, item);
        }
        const auto mapPath = event.value(QStringLiteral("anomaly_map_path")).toString();
        events_->item(row, 0)->setData(Qt::UserRole + 1, mapPath);
        events_->item(row, 1)->setData(Qt::UserRole, id);
        events_->scrollToBottom();
    }

    void showSelectedPreview() {
        const int row = events_->currentRow();
        if (row < 0 || !events_->item(row, 0)) return;
        const auto sourceId = events_->item(row, 1)->data(Qt::UserRole).toString();
        originalImage_ = QPixmap(absolute(sourcePaths_.value(sourceId)));
        if (originalImage_.isNull()) { originalPreview_->setPixmap(QPixmap()); originalPreview_->setText(QStringLiteral("原图不可读取")); }
        else updatePreviewImages();
        const auto mapPath = events_->item(row, 0)->data(Qt::UserRole + 1).toString();
        heatImage_ = QPixmap();
        if (mapPath.isEmpty()) { preview_->setPixmap(QPixmap()); preview_->setText(QStringLiteral("该事件没有保存异常热图")); return; }
        const auto id = events_->item(row, 0)->data(Qt::UserRole).toInt();
        heatImage_ = QPixmap(QDir(activeOutput_).filePath(QStringLiteral("heat-%1.png").arg(id)));
        if (heatImage_.isNull()) { preview_->setPixmap(QPixmap()); preview_->setText(QStringLiteral("热图不可读取；请查看 artifact_error")); return; }
        updatePreviewImages();
    }

    void updatePreviewImages() {
        const auto fit = [](QLabel* label, const QPixmap& image) {
            if (image.isNull()) return;
            const auto available = label->contentsRect().size() - QSize(12, 12);
            if (available.width() > 0 && available.height() > 0)
                label->setPixmap(image.scaled(available, Qt::KeepAspectRatio, Qt::SmoothTransformation));
        };
        fit(originalPreview_, originalImage_);
        fit(preview_, heatImage_);
    }

    void loadSummary() {
        const auto summary = objectFromFile(QDir(activeOutput_).filePath(QStringLiteral("summary.json")));
        if (summary.isEmpty()) { summary_->setText(QStringLiteral("没有完整 summary.json；检查进程日志和 events.jsonl。")); return; }
        const auto latency = summary.value(QStringLiteral("decision_latency_ms")).toObject();
        const QStringList fields{textOf(summary.value(QStringLiteral("generated")), 0), textOf(summary.value(QStringLiteral("completed")), 0),
            textOf(summary.value(QStringLiteral("review")), 0), textOf(summary.value(QStringLiteral("error")), 0),
            textOf(summary.value(QStringLiteral("queue_overflow")), 0), textOf(summary.value(QStringLiteral("timeout")), 0),
            textOf(latency.value(QStringLiteral("p50"))), textOf(latency.value(QStringLiteral("p95"))),
            textOf(latency.value(QStringLiteral("p99"))), summary.value(QStringLiteral("latency_pass")).toBool() ? QStringLiteral("通过") : QStringLiteral("未通过／未声明"),
            textOf(summary.value(QStringLiteral("exit_code")), 0)};
        QString rendered = QStringLiteral("生成 %1 · 完成 %2 · 复检 %3\n错误 %4 · 队列溢出 %5 · 超时 %6\n决策 P50 %7 ms · P95 %8 ms · P99 %9 ms\n延迟门槛 %10 · 退出码 %11");
        for (int i = fields.size(); i > 0; --i) rendered.replace(QStringLiteral("%") + QString::number(i), fields[i - 1]);
        summary_->setText(rendered);
    }

    void appendLog(const QString& text) {
        if (!text.trimmed().isEmpty()) log_->appendPlainText(text.trimmed());
    }

    QDir root_;
    QProcess process_;
    QTimer pollTimer_;
    QString activeOutput_;
    qint64 eventOffset_ = 0;
    int totalEvents_ = 0;
    bool aborted_ = false;
    QHash<QString, QString> sourcePaths_;
    QPixmap originalImage_, heatImage_;
    QComboBox *backend_ = nullptr, *task_ = nullptr, *benchmarkMode_ = nullptr;
    QLineEdit *manifest_ = nullptr, *source_ = nullptr, *output_ = nullptr, *engine_ = nullptr, *evidence_ = nullptr;
    QDoubleSpinBox* fps_ = nullptr;
    QSpinBox *deadline_ = nullptr, *queue_ = nullptr, *warmup_ = nullptr, *duration_ = nullptr, *iterations_ = nullptr;
    QCheckBox *loop_ = nullptr, *cudaPost_ = nullptr;
    QPushButton *start_ = nullptr, *stop_ = nullptr, *openOutput_ = nullptr;
    QLabel *status_ = nullptr, *modelInfo_ = nullptr, *eventCount_ = nullptr, *originalPreview_ = nullptr, *preview_ = nullptr, *summary_ = nullptr;
    QTableWidget* events_ = nullptr;
    QPlainTextEdit* log_ = nullptr;
};

} // namespace

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    InspectionWindow window;
    window.show();
    if (app.arguments().contains(QStringLiteral("--smoke-test")) ||
        app.arguments().contains(QStringLiteral("--smoke-test-compact"))) {
        const bool compact = app.arguments().contains(QStringLiteral("--smoke-test-compact"));
        if (compact) window.resize(960, 700);
        QTimer::singleShot(250, &app, [&app, &window, compact] {
            const auto filename = compact ? QStringLiteral("../reports/gui-preview-compact.png") : QStringLiteral("../reports/gui-preview.png");
            const auto path = QDir(QCoreApplication::applicationDirPath()).absoluteFilePath(filename);
            const bool saved = window.grab().save(path);
            app.exit(saved ? 0 : 4);
        });
    }
    if (app.arguments().contains(QStringLiteral("--smoke-run"))) window.startSmokeRun();
    if (app.arguments().contains(QStringLiteral("--smoke-run-preview"))) window.startSmokeRun(true);
    return app.exec();
}
