---
layout: post
title: yumdownloader를 사용하여 yum 패키지 다운로드
subtitle: 
categories: OS
tags: [Linux, Red Hat]
---

폐쇄망을 사용하는 등의 환경에서 보안 정책 상 yum을 통해 패키지를 설치할 수 없는 경우가 있다.

이 때 외부망 PC에서 yumdownloader를 사용하여 rpm 파일만 다운로드하여 설치할 수 있다.

yumdownloader는 yum-utils 패키지에 포함되어 있다.

따라서 먼저 yum-utils 패키지를 설치해야 한다.

```bash
$ yum install yum-utils
```

yumdownloader로 rpm 파일을 현재 디렉토리에 다운로드한다.

```bash
$ yumdownloader --downloadonly [패키지명]
```

yumdownloader에서 사용할 수 있는 옵션은 아래와 같다.

- `--resolve`
    - 의존성을 가지는 패키지의 rpm 파일까지 모두 다운로드한다.
- `--destdir [디렉토리]`
    - 다운로드할 디렉토리를 직접 지정한다.
- `--source`
    - 컴파일된 바이너리 rpm 대신 소스 rpm을 다운로드한다.

다운로드한 rpm 파일은 `rpm` 명령으로 설치하면 된다.

한 디렉토리에 의존성을 가지는 rpm 파일까지 모두 다운로드하고 아래와 같이 전체 설치하는 방법을 추천한다.

```bash
$ for x in *; do rpm -i $x --nodeps; done
```

---

yumdownloader를 사용하지 않더라도 `yum install [패키지명] --downloadonly` 를 사용하면 rpm 파일을 다운로드할 수 있으나,
이 경우 패키지의 기설치여부를 검사하므로 이미 설치된 패키지는 다운로드되지 않는다.
