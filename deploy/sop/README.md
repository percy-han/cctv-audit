# 稽核标准存哪、怎么改版本

改稽核标准**不用改代码、不用重新部署**。标准是 GCS 上的一个 YAML 文件，
按版本号存；客户在 Gemini Enterprise 的 chip 里点哪个版本号，容器就去取哪个文件。

```
gs://<SOP_BUCKET>/<SOP_PREFIX>/<sop_id>.yaml
   默认：gs://study-project-496907-cctv-audit/sop/chagee-store-v1.yaml
```

三个环境变量，都在 `deploy/agent_runtime/deploy.py` 的 `env_vars()` 里：

| 变量 | 现在的值 | 作用 |
| :--- | :--- | :--- |
| `SOP_BUCKET` | `study-project-496907-cctv-audit` | 桶 |
| `SOP_PREFIX` | `sop` | 桶里的目录 |
| `DEFAULT_SOP_ID` | （空） | 客户没说版本号时用哪个。**留空更安全**，见下 |

## 版本号怎么起

`<客户>-<场景>-v<数字>`，比如 `chagee-store-v1`、`chagee-drinkmaking-v2`。

只允许字母、数字、点、下划线、连字符，最长 64 个字符
（`analyzer/sop.py` 里的 `_SOP_ID_RE`）。这不是洁癖：版本号是客户在聊天框里
打出来的字，然后被拼进一个对象路径，不卡住形状就能拼出 `../` 走到别的目录去。

## 三条硬规矩

**一、已经发布的版本不再改内容。** 每条稽核记录上都盖着
`sop_id` 和 `sop_version`。如果 `v1` 的内容事后能改，「这条违规是按 v1 判的」
这句话就不再说明任何事，**所有历史报告一起失去可核对性**。
改标准 = 发一个新版本号，老的留着。

`publish.py` 会挡住覆盖，要覆盖得显式加 `--force`。

**二、取不到就报错，绝不回落。** 客户点了 `chagee-store-v3` 而桶里没有，
容器直接拒单并把版本号写在拒绝理由里，不会偷偷拿别的版本去判。
理由是：拿错版本判出来的报告，和真的一模一样，下游没有任何人看得出来。

**三、`DEFAULT_SOP_ID` 尽量留空。** 配了它，客户不说版本号也能跑，
跑出来的报告按的是部署时定的那版——这是「替客户选版本」，
只有在客户确认「我们只有一套标准」的时候才配。

## 加一个新版本，三步

```bash
# 1. 拿现在这版做底子改
python deploy/sop/publish.py show chagee-store-v1 > /tmp/v2.yaml
$EDITOR /tmp/v2.yaml

# 2. 先看看模型实际会读到什么（不上传）
python deploy/sop/publish.py render chagee-store-v2 /tmp/v2.yaml

# 3. 发布
python deploy/sop/publish.py put chagee-store-v2 /tmp/v2.yaml
```

`render` 那步值得每次都做。YAML 里的 `description` 是**原样进 prompt** 的，
改一句话的措辞就是改判定口径，而这件事看 YAML 看不出来，看渲染结果才看得出来。

第 3 步之后，去 `deploy/ge/register.py` 把 chip 的文案里的版本号改掉，
重新 `update` 一次 agent。chip 只有 `text` 一个字段、藏不了参数，
所以**版本号必须写在客户看得见的句子里**——客户能改它，
这正是上面第二条「取不到就报错」必须成立的原因。

## YAML 长什么样

字段说明写在 `cctv_audit/analyzer/sop_rules.yaml` 的文件头注释里，
那份也是本地 `adk web` 跑的时候用的标准（`SOP_RULES_PATH`）。
两个字段最容易写错，都在 `analyzer/sop.py` 里有对应逻辑：

- **`detection_type: presence | absence`** —— `presence` 是「看到就违规」，
  举证容易；`absence` 是「整段都没看到某个必做动作才算违规」，举证困难，
  容易假阳性。拿不准就用 `presence`。
- **`requires_full_context: true`** —— 这条依赖完整过程。加上它，
  人走出画面、动作跨窗口边界的时候模型判 `CANNOT_DETERMINE` 而不是 `VIOLATION`。
  「洗手了没有」这类必须加。

## 谁能改

发布需要对桶的写权限。跑稽核的服务账号
`cctv-audit-agent@study-project-496907.iam.gserviceaccount.com` 有
`roles/storage.objectAdmin`；改标准的人用自己的账号即可，
不需要也不应该拿服务账号的凭据。
